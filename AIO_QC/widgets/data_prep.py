# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Preparation (QC drawer ▸ Dataset; Ctrl+Alt+P) — get raw data ready before it is opened as a dataset, or
grow a dataset you have. Works without a dataset open.

Every tool works the same way: choose folders and options → Preview (a plan: counts and the files concerned —
nothing changes) → Run (in the background, stoppable) → a summary, the log (CSV) and Undo. Runs are logged in
<folder>/_aioqc_prep/; "Undo a run" lists them for any folder.

    Unzip archives      every .zip under a folder (nested too), zip-slip / zip-bomb guarded, corrupt ones reported
    Combine & flatten   nested folders (several sources) -> one flat LabelMe folder: smart pairing, images restored
                        from imageData, imageData → null, exact duplicates set aside with their JSON (labels compared),
                        orphans sorted, name clashes renamed
    Find orphans        in place: JSONs without image / images without JSON moved to sibling folders
    Integrity check     full decode of every image, LabelMe sanity of every JSON -> quarantine (with partners)
    Exact duplicates    byte-identical images in a folder tree, best copy kept, the rest (+ JSON) set aside
    Resize images       downscale big images into a new folder; LabelMe points / sizes scaled; YOLO labels copied
    Import annotations  COCO / Pascal VOC / CVAT / Label Studio -> a new LabelMe or YOLO folder
    Fix EXIF rotation   rotation tags baked into the pixels, labels brought into the same upright frame
    Pick images to label  the most varied (and, with a model, most uncertain) images of an unlabelled pool
"""
from __future__ import annotations

import os
import re
import threading
import types
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QSpinBox, QSplitter,
    QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core.prep import common as C
from AIO_QC.core.prep import exif_fix as X
from AIO_QC.core.prep import flatten as F
from AIO_QC.core.prep import importers as IM
from AIO_QC.core.prep import pick as PK
from AIO_QC.core.prep import tasks as T
from AIO_QC.utils import newButton, newIcon, style as apply_style
from . import anim
from AIO_QC.widgets.lifecycle import RetainedThread
from AIO_QC.core.diag import ignored as _ignored


class Job(RetainedThread, QThread):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self.cancel, lambda d, t, label="": self.progress.emit(int(d), int(t), label)))
        except C.Cancelled:
            self.failed.emit("stopped")
        except Exception as exc:
            logger.exception("data prep")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


_READS = ("text", "isChecked", "value", "currentData", "currentText", "toPlainText", "currentIndex")


class _FrozenItem:
    def __init__(self, it):
        self._text, self._data = it.text(), {r: it.data(r) for r in (Qt.ItemDataRole.UserRole,)}

    def text(self):
        return self._text

    def data(self, role):
        return self._data.get(role)


class _Frozen:
    """Values of one input widget, read on the UI thread, readable from a worker (no Qt calls)."""

    def __init__(self, w):
        self._v = {}
        for name in _READS:
            f = getattr(w, name, None)
            if callable(f):
                try:
                    self._v[name] = f()
                except TypeError:
                    pass
        if isinstance(w, QListWidget):
            it = w.currentItem()
            self._v["currentItem"] = None if it is None else _FrozenItem(it)
        for name, val in self._v.items():
            setattr(self, name, (lambda v: lambda *_a: v)(val))


_INPUTS = (QLineEdit, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget)


def _plain(m) -> str:
    from .errors import message
    return message(m)


class PageSnapshot:
    """A page whose input widgets are frozen copies: make_plan / run execute in a worker thread and must never
    touch a live widget (Qt objects are not thread-safe). Built on the UI thread right before the job starts."""

    def __init__(self, page):
        self.__dict__["_page"] = page
        self.__dict__["_vals"] = {k: _Frozen(v) for k, v in vars(page).items() if isinstance(v, _INPUTS)}

    def __getattr__(self, k):
        vals = self.__dict__["_vals"]
        if k in vals:
            return vals[k]
        page = self.__dict__["_page"]
        f = getattr(type(page), k, None)
        if isinstance(f, types.FunctionType):
            return types.MethodType(f, self)
        return getattr(page, k)

    def __setattr__(self, k, v):
        setattr(self.__dict__["_page"], k, v)


def _folder_row(parent, text, setting_key, settings, save=False, tip=""):
    edit = QLineEdit(settings.value(setting_key, "") or "")
    edit.setPlaceholderText(tip or "Folder")
    edit.textChanged.connect(lambda t: settings.setValue(setting_key, t))
    b = newButton("…", slot=lambda: _pick(parent, edit, text))
    b.setFixedWidth(32)
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    lay.addWidget(b)
    return w, edit


def _pick(parent, edit, title):
    d = QFileDialog.getExistingDirectory(parent, title, edit.text())
    if d:
        edit.setText(d)


def _naming_box(page, context):
    """The naming panel (settings only) for files that would share a name; page._naming = its scheme (a plain dict,
    read by the worker thread)."""
    from .naming_panel import NamingPanel
    panel = NamingPanel(page, context=f"prep_{context}", allow_scope=True, preview=False)
    panel.setToolTip("Files that would get the same name in the flat folder (camA/img001 and camB/img001): how they "
                     "are renamed. The plan checks every name against the others and the folder's files.")

    def keep(_p=None):
        page._naming = panel.scheme().to_dict()
    panel.planChanged.connect(keep)
    keep()
    return panel


# ====================================================================================================== pages
class Page(QWidget):
    TITLE = ""
    ICON = "open"
    HELP = ""
    RUN = "Run"

    def __init__(self, dlg):
        super().__init__()
        self.dlg, self.st = dlg, dlg.app.settings
        self.form = QFormLayout()
        self.build()
        help_ = QLabel(self.HELP)
        help_.setWordWrap(True)
        apply_style(help_, "color:@muted;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel(self.TITLE)
        apply_style(title, "font-size:14pt; font-weight:600;")
        lay.addWidget(title)
        lay.addWidget(help_)
        lay.addLayout(self.form)
        lay.addStretch(1)                  # the pages share one height: the spare room goes below the form

    def cb(self, text, key, default, tip=""):
        c = QCheckBox(text)
        c.setChecked(self.st.value(f"prep/{key}", default, type=bool))
        c.toggled.connect(lambda on: self.st.setValue(f"prep/{key}", on))
        c.setToolTip(tip)
        return c

    def folder(self, label, key, tip=""):
        w, e = _folder_row(self, label, f"prep/{key}", self.st, tip=tip)
        self.form.addRow(label, w)
        return e

    def need(self, *edits):
        for e in edits:
            p = e.text().strip()
            if not p or not Path(p).is_dir():
                raise ValueError(f"Choose an existing folder ({p or 'empty'}).")

    # subclasses
    def build(self):
        ...

    def make_plan(self, cancel, progress):
        ...

    def run(self, plan, cancel, progress):
        ...

    def result_lines(self, st) -> list[str]:
        lines = []
        for k in ("done", "extracted", "files", "resized"):
            if k in st and not isinstance(st[k], dict):
                lines.append(f"{k}: {st[k]:,}")
        return lines

    def out_folder(self, plan, st):
        return None

    def dataset_folder(self, plan, st):
        return None


class UnzipPage(Page):
    TITLE, ICON = "Unzip archives", "import"
    HELP = ("Every .zip under the folder is extracted into a folder named after it (nested zips too). Archives already "
            "extracted are skipped; corrupt, password-protected and unsafe archives (paths escaping the folder, zip "
            "bombs) are reported and left alone.")

    def build(self):
        self.root = self.folder("Folder:", "unzip_root")
        self.nested = self.cb("Also extract zips found inside the archives", "unzip_nested", True)
        self.move = self.cb("Move each extracted archive to _extracted_zips/", "unzip_move", False)
        self.form.addRow(self.nested)
        self.form.addRow(self.move)

    def make_plan(self, cancel, progress):
        self.need(self.root)
        return T.unzip_plan(self.root.text().strip(), self.nested.isChecked(), progress, cancel)

    def run(self, plan, cancel, progress):
        return T.unzip_run(plan, self.move.isChecked(), progress, cancel)

    def result_lines(self, st):
        return [f"Extracted {st['extracted']:,} archive(s) · {st['files']:,} files · {C.human(st['bytes'])}"
                + (f" · {st['nested']} nested" if st["nested"] else "")]

    def out_folder(self, plan, st):
        return plan.opts["root"]


class FlattenPage(Page):
    TITLE, ICON = "Combine & flatten", "tool-export"
    HELP = ("Nested folders (one or several) of images + LabelMe JSONs → one flat folder. JSONs are paired with "
            "their image (imagePath, same name in the folder, or a unique name anywhere); images that exist only as "
            "imageData are restored; byte-identical images are added once — the other copies go to "
            "orphans/duplicates WITH their JSON (the plan lists copies whose labels differ); images without JSON and "
            "JSONs without image are sorted into orphans/. Name clashes of different images get _1, _2 ….")

    def build(self):
        self.sources = QPlainTextEdit(self.st.value("prep/flat_sources", "") or "")
        self.sources.setPlaceholderText("One source folder per line (or Add…)")
        self.sources.setFixedHeight(70)
        self.sources.textChanged.connect(lambda: self.st.setValue("prep/flat_sources", self.sources.toPlainText()))
        add = newButton("Add…", icon="open", slot=self._add)
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.sources, 1)
        lay.addWidget(add, 0, Qt.AlignmentFlag.AlignTop)
        self.form.addRow("Sources:", w)
        self.dst = self.folder("Flat folder:", "flat_dst", "New or existing folder (its images are compared too)")
        self.orph = self.folder("Orphans folder:", "flat_orph", "Default: <flat folder>_orphans")
        self.mode = QComboBox()
        self.mode.addItem("Copy (sources untouched)", True)
        self.mode.addItem("Move (sources emptied; empty folders removed)", False)
        self.mode.setCurrentIndex(0 if self.st.value("prep/flat_copy", True, type=bool) else 1)
        self.mode.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/flat_copy", self.mode.currentData()))
        self.form.addRow("Mode:", self.mode)
        self.null = self.cb("imageData → null (small JSONs; the image file is the source)", "flat_null", True)
        self.restore = self.cb("Restore images that exist only as imageData", "flat_restore", True)
        self.dedupe = self.cb("Add exact duplicate images only once", "flat_dedupe", True)
        self.bg = self.cb("Images without JSON go to the flat folder (background / negatives)", "flat_bg", False)
        for c in (self.null, self.restore, self.dedupe, self.bg):
            self.form.addRow(c)
        self.naming = _naming_box(self, "flatten")
        self.form.addRow("Same names:", self.naming)

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Source folder")
        if d:
            t = self.sources.toPlainText().rstrip()
            self.sources.setPlainText((t + "\n" if t else "") + d)

    def make_plan(self, cancel, progress):
        srcs = [s.strip().strip('"') for s in self.sources.toPlainText().splitlines() if s.strip()]
        if not srcs:
            raise ValueError("Add at least one source folder.")
        for s in srcs:
            if not Path(s).is_dir():
                raise ValueError(f"Not a folder: {s}")
        if not self.dst.text().strip():
            raise ValueError("Choose the flat folder.")
        o = F.Options(srcs, self.dst.text().strip(), self.orph.text().strip(), copy=self.mode.currentData(),
                      null_image_data=self.null.isChecked(), restore_images=self.restore.isChecked(),
                      dedupe=self.dedupe.isChecked(), background_images=self.bg.isChecked(),
                      naming=dict(self._naming))
        P = F.plan(o, progress, cancel)
        P.rows = P.summary_rows()
        return P

    def run(self, plan, cancel, progress):
        return F.run(plan, progress, cancel)

    def result_lines(self, st):
        d = st["done"]
        out = [f"{d.get('pair', 0):,} pairs · {d.get('restore', 0):,} restored · {d.get('dup', 0):,} duplicates · "
               f"{d.get('orphan_img', 0):,} orphan images · {d.get('orphan_json', 0):,} orphan JSONs"]
        b = st.get("json_bytes") or (0, 0)
        if b[0]:
            out.append(f"Rewritten JSONs: {C.human(b[0])} → {C.human(b[1])}")
        if st.get("removed_dirs"):
            out.append(f"{st['removed_dirs']} empty source folder(s) removed")
        return out

    def out_folder(self, plan, st):
        return plan.opts.dst

    def dataset_folder(self, plan, st):
        return plan.opts.dst


class CombineYoloPage(Page):
    TITLE, ICON = "Combine YOLO datasets", "tool-export"
    HELP = ("One or several YOLO datasets (images/ and labels/ mirrored, with or without train / val / test, any "
            "nesting) → one dataset with one images/ and labels/ folder per split. The class lists are merged by "
            "name (letter case and spaces ignored; the first dataset's order kept, new classes added at the end) "
            "and every label file gets the new ids. Files that would share a name are renamed with the naming "
            "scheme below (every name checked; renamed.csv lists them); exact duplicate images are added once. "
            "Hard links where possible — the sources are never changed; Undo removes what the run created.")

    def build(self):
        self.sources = QPlainTextEdit(self.st.value("prep/yc_sources", "") or "")
        self.sources.setPlaceholderText("One YOLO dataset folder per line (or Add…)")
        self.sources.setFixedHeight(70)
        self.sources.textChanged.connect(lambda: self.st.setValue("prep/yc_sources", self.sources.toPlainText()))
        add = newButton("Add…", icon="open", slot=self._add)
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.sources, 1)
        lay.addWidget(add, 0, Qt.AlignmentFlag.AlignTop)
        self.form.addRow("Datasets:", w)
        self.dst = self.folder("Combined folder:", "yc_dst", "A new folder (outside the datasets)")
        self.link = QComboBox()
        self.link.addItem("Hard links (no extra disk space; copies when not possible)", "hardlink")
        self.link.addItem("Copies", "copy")
        self.form.addRow("Images:", self.link)
        self.splits = self.cb("Keep train / val / test (else everything in one images/ folder)", "yc_splits", True)
        self.dedupe = self.cb("Add exact duplicate images only once", "yc_dedupe", True)
        self.bg = self.cb("Keep images without a label file (background: an empty label)", "yc_bg", True)
        for c in (self.splits, self.dedupe, self.bg):
            self.form.addRow(c)
        self.naming = _naming_box(self, "combine_yolo")
        self.form.addRow("Same names:", self.naming)

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "YOLO dataset folder")
        if d:
            t = self.sources.toPlainText().rstrip()
            self.sources.setPlainText((t + "\n" if t else "") + d)

    def make_plan(self, cancel, progress):
        from AIO_QC.core.prep import yolo_combine as YCB
        srcs = [s.strip().strip('"') for s in self.sources.toPlainText().splitlines() if s.strip()]
        if not srcs:
            raise ValueError("Add at least one YOLO dataset folder.")
        for s in srcs:
            if not Path(s).is_dir():
                raise ValueError(f"Not a folder: {s}")
        if not self.dst.text().strip():
            raise ValueError("Choose the combined folder.")
        o = YCB.Options(srcs, self.dst.text().strip(), keep_splits=self.splits.isChecked(),
                        link=self.link.currentData(), dedupe=self.dedupe.isChecked(), background=self.bg.isChecked(),
                        naming=dict(self._naming))
        P = YCB.plan(o, progress, cancel)
        P.rows = P.summary_rows()
        return P

    def run(self, plan, cancel, progress):
        from AIO_QC.core.prep import yolo_combine as YCB
        return YCB.run(plan, progress, cancel)

    def result_lines(self, st):
        d = st["done"]
        return [f"{d.get('images', 0):,} images · {d.get('link', 0):,} linked · {d.get('copy', 0):,} copied"]

    def out_folder(self, plan, st):
        return plan.opts.dst

    def dataset_folder(self, plan, st):
        return plan.opts.dst


class OrphansPage(Page):
    TITLE, ICON = "Find orphans", "tool-empty"
    HELP = ("In place: pairs stay where they are; JSONs without an image move to jsons_without_images/ and images "
            "without a JSON to images_without_jsons/ (sub-folders kept). Pairing: same name in the same folder, "
            "else the JSON's imagePath.")

    def build(self):
        self.root = self.folder("Folder:", "orph_root")
        self.out = self.folder("Put the orphan folders in:", "orph_out", "Default: the folder itself")
        self.ip = self.cb("Also pair through the JSON's imagePath", "orph_ip", True)
        self.copy = self.cb("Copy instead of move", "orph_copy", False)
        self.form.addRow(self.ip)
        self.form.addRow(self.copy)

    def make_plan(self, cancel, progress):
        self.need(self.root)
        return T.orphans_plan(self.root.text().strip(), self.out.text().strip() or None, self.ip.isChecked(),
                              progress, cancel)

    def run(self, plan, cancel, progress):
        return T.move_run(plan, plan.opts["out"], self.copy.isChecked(), progress, cancel)

    def result_lines(self, st):
        return [f"{st['done']:,} orphan file(s) {'copied' if self.copy.isChecked() else 'moved'}"]

    def out_folder(self, plan, st):
        return plan.opts["out"]

    def dataset_folder(self, plan, st):
        return plan.opts["root"]


class IntegrityPage(Page):
    TITLE, ICON = "Integrity check", "tool-quality"
    HELP = ("Every image is decoded completely (truncated JPEGs, 0-byte files, wrong file types …) and every JSON is "
            "checked as LabelMe (valid JSON, a shapes list, numeric points, labels, imageWidth / imageHeight equal to "
            "the real image, an image or imageData to go with it). Bad files move to a quarantine folder, structure "
            "kept — with their partner (image / JSON of the same name) if you like.")

    def build(self):
        self.root = self.folder("Folder:", "int_root")
        self.q = self.folder("Quarantine folder:", "int_q", "Default: <folder>_quarantine")
        self.imgs = self.cb("Check images", "int_imgs", True)
        self.jsons = self.cb("Check JSONs", "int_jsons", True)
        self.partner = self.cb("Quarantine the partner file too (keeps pairs together)", "int_partner", True)
        for c in (self.imgs, self.jsons, self.partner):
            self.form.addRow(c)

    def make_plan(self, cancel, progress):
        self.need(self.root)
        return T.integrity_plan(self.root.text().strip(), self.q.text().strip() or None, self.imgs.isChecked(),
                                self.jsons.isChecked(), self.partner.isChecked(), progress, cancel)

    def run(self, plan, cancel, progress):
        return T.move_run(plan, plan.opts["root"], False, progress, cancel)

    def result_lines(self, st):
        return [f"{st['done']:,} file(s) moved to quarantine"]

    def out_folder(self, plan, st):
        return plan.opts["quarantine"]

    def dataset_folder(self, plan, st):
        return plan.opts["root"]


class DupesPage(Page):
    TITLE, ICON = "Exact duplicates", "tool-image-dups"
    HELP = ("Byte-identical images anywhere in the folder tree, whatever their names (size first, then a quick "
            "fingerprint, then SHA-256 — only collisions are read). The copy kept: annotated > original name "
            "(fewest _1 / _2) > oldest > A–Z. The others go aside WITH their JSON; copies whose labels DIFFER from "
            "the kept one are listed for review. (Near-duplicates: open the folder and use Dataset ▸ Duplicate "
            "Images.)")

    def build(self):
        self.root = self.folder("Folder:", "dup_root")
        self.dd = self.folder("Move duplicates to:", "dup_dir", "Default: <folder>_duplicates")

    def make_plan(self, cancel, progress):
        self.need(self.root)
        return T.dupes_plan(self.root.text().strip(), self.dd.text().strip() or None, progress, cancel)

    def run(self, plan, cancel, progress):
        return T.move_run(plan, plan.opts["root"], False, progress, cancel)

    def result_lines(self, st):
        return [f"{st['done']:,} file(s) set aside"]

    def out_folder(self, plan, st):
        return plan.opts["dup_dir"]

    def dataset_folder(self, plan, st):
        return plan.opts["root"]


class ResizePage(Page):
    TITLE, ICON = "Resize images", "fit-window"
    HELP = ("Downscale large images into a NEW folder with the same structure (sources are never touched). LabelMe "
            "JSONs are scaled with their image (points, imageWidth / imageHeight; imageData dropped); YOLO labels are "
            "normalised and copied as they are. EXIF rotation is applied like the viewer does.")

    def build(self):
        self.src = self.folder("Source folder:", "rs_src")
        self.dst = self.folder("Output folder:", "rs_dst")
        self.mode = QComboBox()
        for t, k in (("Longest side at most …", "long"), ("Images at least … tall → × factor", "height"),
                     ("Every image × factor", "factor")):
            self.mode.addItem(t, k)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.st.value("prep/rs_mode", "long"))))
        self.mode.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/rs_mode", self.mode.currentData()))
        self.limit = QSpinBox()
        self.limit.setRange(64, 20000)
        self.limit.setSuffix(" px")
        self.limit.setValue(int(self.st.value("prep/rs_limit", 1920)))
        self.limit.valueChanged.connect(lambda v: self.st.setValue("prep/rs_limit", v))
        self.factor = QDoubleSpinBox()
        self.factor.setRange(0.05, 1.0)
        self.factor.setSingleStep(0.05)
        self.factor.setValue(float(self.st.value("prep/rs_factor", 0.5)))
        self.factor.valueChanged.connect(lambda v: self.st.setValue("prep/rs_factor", v))
        self.fmt = QComboBox()
        for t, k in (("Keep the format", "keep"), ("JPEG", "jpg"), ("PNG", "png")):
            self.fmt.addItem(t, k)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(self.st.value("prep/rs_fmt", "keep"))))
        self.fmt.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/rs_fmt", self.fmt.currentData()))
        self.q = QSpinBox()
        self.q.setRange(30, 100)
        self.q.setValue(int(self.st.value("prep/rs_q", 90)))
        self.q.valueChanged.connect(lambda v: self.st.setValue("prep/rs_q", v))
        self.copy_small = self.cb("Images that need no resize: copy unchanged (no re-encode)", "rs_copy", True)
        self.ann = self.cb("Bring the annotations (LabelMe scaled, YOLO / data.yaml copied)", "rs_ann", True)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        for w in (self.mode, self.limit, QLabel("factor"), self.factor):
            rl.addWidget(w)
        self.form.addRow("Rule:", row)
        row2 = QWidget()
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.addWidget(self.fmt)
        r2.addWidget(QLabel("quality"))
        r2.addWidget(self.q)
        r2.addStretch(1)
        self.form.addRow("Format:", row2)
        self.form.addRow(self.copy_small)
        self.form.addRow(self.ann)

    def make_plan(self, cancel, progress):
        self.need(self.src)
        if not self.dst.text().strip():
            raise ValueError("Choose the output folder.")
        o = T.ResizeOptions(self.src.text().strip(), self.dst.text().strip(), self.mode.currentData(),
                            self.limit.value(), self.factor.value(), self.fmt.currentData(), self.q.value(),
                            self.copy_small.isChecked(), self.ann.isChecked())
        return T.resize_plan(o, progress, cancel)

    def run(self, plan, cancel, progress):
        return T.resize_run(plan, progress, cancel)

    def result_lines(self, st):
        b = st["bytes"]
        return [f"{st['done']:,} file(s) written · {st['resized']:,} image(s) downscaled · images "
                f"{C.human(b[0])} → {C.human(b[1])}"]

    def out_folder(self, plan, st):
        return plan.opts["o"].dst

    def dataset_folder(self, plan, st):
        return plan.opts["o"].dst


class ImportPage(Page):
    TITLE, ICON = "Import annotations", "import"
    HELP = ("Annotations from another tool → a NEW folder as LabelMe, YOLO or a classification dataset (the sources are "
            "not touched). Main formats: COCO JSON (boxes, polygons, RLE masks, keypoints grouped with their box), "
            "segmentation masks (a folder of PNGs: class index, colour or instance), Pascal VOC XML (a folder) and "
            "CVAT XML (images and video tracks; whole-image tags). Also: Label Studio JSON (boxes incl. rotated, "
            "polygons, points, ellipses, brush masks, choices), KITTI label .txt, VGG Image Annotator JSON, a "
            "Supervisely project folder, CreateML JSON, Open Images CSV, Datumaro JSON and classification tables "
            "(CSV / TSV / TXT: file + class, several classes in one cell, one 1 / 0 column per class, or "
            "“path label” lists). The format is detected by itself. Images are matched by path, then by name, then "
            "by name without the extension; they are hard-linked (no extra space) or copied.")

    def build(self):
        self.src = QLineEdit(self.st.value("prep/im_src", "") or "")
        self.src.setPlaceholderText("A .json / .xml / .csv file, or a folder (VOC .xml, PNG masks, KITTI .txt, a "
                                    "Supervisely project)")
        self.src.textChanged.connect(lambda t: (self.st.setValue("prep/im_src", t), self._guess()))
        bf = newButton("File…", slot=self._pick_file)
        bd = newButton("Folder…", slot=lambda: _pick(self, self.src, "Folder of annotations (VOC, masks, KITTI, "
                                                                      "Supervisely)"))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.src, 1)
        h.addWidget(bf)
        h.addWidget(bd)
        self.form.addRow("Annotations:", row)
        self.images = self.folder("Images folder:", "im_images", "The folder that holds the images (searched with "
                                                                  "its sub-folders)")
        self.out = self.folder("Output folder:", "im_out", "A new or empty folder")
        self.fmt = QComboBox()
        self.fmt.addItem("Detect automatically", "auto")
        for k, t in IM.FORMATS.items():
            self.fmt.addItem(t, k)
        self.fmt.currentIndexChanged.connect(lambda _i: self._guess())
        self.detected = QLabel("")
        apply_style(self.detected, "color:@muted;")
        frow = QWidget()
        fh = QHBoxLayout(frow)
        fh.setContentsMargins(0, 0, 0, 0)
        fh.addWidget(self.fmt)
        fh.addWidget(self.detected, 1)
        self.form.addRow("Format:", frow)
        self.output = QComboBox()
        for k, t in IM.OUTPUTS.items():
            self.output.addItem(t, k)
        self.output.setCurrentIndex(max(0, self.output.findData(self.st.value("prep/im_output", "labelme"))))
        self.output.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/im_output", self.output.currentData()))
        self.form.addRow("Write as:", self.output)
        self.masks = self.cb("Polygons from segmentations / masks (otherwise boxes only)", "im_masks", True)
        self.link = self.cb("Hard-link the images (no extra space; copied where linking is not possible)", "im_link",
                            True)
        self.form.addRow(self.masks)
        self.form.addRow(self.link)
        self.via_attr = QComboBox()
        self.via_attr.setToolTip("VGG Image Annotator keeps the class in a region attribute — choose which one")
        self.via_lbl = QLabel("Class from attribute:")
        self.form.addRow(self.via_lbl, self.via_attr)
        self.via_attr.setVisible(False)
        self.via_lbl.setVisible(False)
        self._build_csv()
        self._build_masks()
        self._guess()

    # ---------------------------------------------------------------- classification tables
    def _build_csv(self):
        self._spec = None
        self.csv_box = QGroupBox("Classification table")
        g = QFormLayout(self.csv_box)
        self.csv_desc = QLabel("")
        self.csv_desc.setWordWrap(True)
        g.addRow(self.csv_desc)
        self.c_file, self.c_label, self.c_split = QComboBox(), QComboBox(), QComboBox()
        self.c_mode = QComboBox()
        self.c_mode.addItem("One column with the class(es)", "column")
        self.c_mode.addItem("One column per class (1 / 0)", "onehot")
        self.c_sep = QComboBox()
        for t, k in (("One class per cell", "none"), ("Several, split by ;", ";"), ("Several, split by |", "|"),
                     ("Several, split by ,", ","), ("Several, split by spaces", "space"), ("Find out", "auto")):
            self.c_sep.addItem(t, k)
        self.c_names = QLineEdit()
        self.c_names.setPlaceholderText("Optional: classes.txt / data.yaml — when the classes are numbers")
        g.addRow("Image file in", self.c_file)
        g.addRow("Classes", self.c_mode)
        g.addRow("Class column", self.c_label)
        g.addRow("Several per image", self.c_sep)
        g.addRow("Split column", self.c_split)
        g.addRow("Class names", self.c_names)
        self.c_preview = QLabel("")
        self.c_preview.setWordWrap(True)
        apply_style(self.c_preview, "color:@muted;")
        g.addRow(self.c_preview)
        for w in (self.c_file, self.c_label, self.c_split, self.c_mode, self.c_sep):
            w.currentIndexChanged.connect(lambda _i: self._csv_spec())
        self.c_names.textChanged.connect(lambda _t: self._csv_spec())
        self.form.addRow(self.csv_box)
        self.csv_box.setVisible(False)

    def _build_masks(self):
        self.mask_box = QGroupBox("Segmentation masks")
        g = QFormLayout(self.mask_box)
        self.m_desc = QLabel("")
        self.m_desc.setWordWrap(True)
        g.addRow(self.m_desc)
        self.m_mode = QComboBox()
        for t, k in (("Find out", "auto"), ("Class index (0 = background, 1 … = classes)", "index"),
                     ("Colour per class (RGB)", "color"), ("One value per object (instances)", "instance")):
            self.m_mode.addItem(t, k)
        self.m_names = QLineEdit()
        self.m_names.setPlaceholderText("labelmap.txt (name:r,g,b) or classes.txt — class 1 = the first name "
                                        "(“background” first is skipped)")
        self.m_class = QLineEdit()
        self.m_class.setPlaceholderText("Instance masks: the folder of class masks that says each object's class")
        self.m_min = QSpinBox()
        self.m_min.setRange(0, 100000)
        self.m_min.setValue(16)
        self.m_min.setSuffix(" px²")
        g.addRow("Pixel values", self.m_mode)
        g.addRow("Class names", self.m_names)
        g.addRow("Class masks", self.m_class)
        g.addRow("Leave out regions under", self.m_min)
        self.form.addRow(self.mask_box)
        self.mask_box.setVisible(False)

    def _mask_spec(self) -> dict:
        return {"mode": self.m_mode.currentData(), "labelmap": self.m_names.text().strip(),
                "class_masks": self.m_class.text().strip(), "min_area": self.m_min.value()}

    def _csv_load(self, path):
        from AIO_QC.core.prep import cls_csv as CC
        try:
            spec, head, rows, desc = CC.sniff(path)
        except Exception as exc:
            self.csv_desc.setText(f"Could not read the table: {exc}")
            return
        self._head = head
        for combo in (self.c_file, self.c_label, self.c_split):
            combo.blockSignals(True)
            combo.clear()
        self.c_split.addItem("(none)", -1)
        for i, h in enumerate(head):
            self.c_file.addItem(h, i)
            self.c_label.addItem(h, i)
            self.c_split.addItem(h, i)
        self.c_file.setCurrentIndex(spec.file_col)
        self.c_label.setCurrentIndex(spec.label_col if spec.mode == "column" else 0)
        self.c_split.setCurrentIndex(spec.split_col + 1)
        self.c_mode.setCurrentIndex(self.c_mode.findData(spec.mode))
        self.c_sep.setCurrentIndex(max(0, self.c_sep.findData(spec.sep)))
        for combo in (self.c_file, self.c_label, self.c_split):
            combo.blockSignals(False)
        self._base_spec = spec
        self.csv_desc.setText(f"Found: {desc}")
        sample = "; ".join(", ".join(r[:4]) for r in rows[:3])
        self.c_preview.setText(f"First rows: {sample[:300]}")
        self._csv_spec()
        if self.output.currentData() not in IM.CLS_OUTPUTS:
            self.output.setCurrentIndex(self.output.findData("cls_folders"))

    def _csv_spec(self):
        from AIO_QC.core.prep import cls_csv as CC
        base = getattr(self, "_base_spec", None)
        if base is None:
            return
        mode = self.c_mode.currentData()
        fc = self.c_file.currentData() or 0
        sc = self.c_split.currentData()
        onehot = [i for i in (base.onehot_cols or range(len(self._head))) if i not in (fc, sc)] \
            if mode == "onehot" else []
        self._spec = CC.Spec(file_col=fc, mode=mode, label_col=self.c_label.currentData() or 0, onehot_cols=onehot,
                             sep=self.c_sep.currentData(), split_col=-1 if sc is None else sc,
                             names=CC.read_names(self.c_names.text().strip()), delimiter=base.delimiter,
                             header=base.header)
        self.c_label.setEnabled(mode == "column")
        self.c_sep.setEnabled(mode == "column")

    def _pick_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Annotation file", self.src.text(),
                                           "Annotations (*.json *.jsonl *.xml *.csv *.tsv *.txt);;All files (*)")
        if p:
            self.src.setText(p)

    def _guess(self):
        p = self.src.text().strip()
        try:
            f = IM.detect(p) if p and Path(p).exists() else None
        except Exception as _exc:
            _ignored(_exc)
            f = None
        self.detected.setText(f"detected: {IM.FORMATS[f]}" if f else ("" if not p else "format not recognised"))
        csv = (f == "csv" and self.fmt.currentData() in ("auto", "csv")) or self.fmt.currentData() == "csv"
        if hasattr(self, "csv_box"):
            self.csv_box.setVisible(bool(csv and p))
            if csv and p and Path(p).is_file():
                self._csv_load(p)
        via = (f == "via" and self.fmt.currentData() in ("auto", "via")) or self.fmt.currentData() == "via"
        if hasattr(self, "via_attr"):
            self.via_attr.setVisible(bool(via and p))
            self.via_lbl.setVisible(bool(via and p))
            if via and p and Path(p).is_file():
                from AIO_QC.core.prep.readers_more import via_attributes
                keep = self.via_attr.currentText()
                self.via_attr.clear()
                try:
                    names = via_attributes(p)
                except Exception as _exc:
                    _ignored(_exc)
                    names = []
                self.via_attr.addItems(names or ["(first attribute with a value)"])
                if keep in names:
                    self.via_attr.setCurrentText(keep)
        masks = (f == "masks" and self.fmt.currentData() in ("auto", "masks")) or self.fmt.currentData() == "masks"
        if hasattr(self, "mask_box"):
            self.mask_box.setVisible(bool(masks and p))
            if masks and p and Path(p).is_dir():
                try:
                    from AIO_QC.core.prep.masks import sniff
                    info = sniff(p)
                    vals = sorted(info["values"])
                    self.m_desc.setText(f"{info['n']:,} mask(s) · looks like: {info['mode']}" +
                                        (f" · values {vals[:12]}{' …' if len(vals) > 12 else ''}" if vals else "") +
                                        (f" · {info['colors']} colours" if info["colors"] else ""))
                except Exception as exc:
                    self.m_desc.setText(f"Could not read the masks: {exc}")
                if self.output.currentData() not in ("labelme", "yolo_seg"):
                    self.output.setCurrentIndex(self.output.findData("yolo_seg"))

    def make_plan(self, cancel, progress):
        p = self.src.text().strip()
        if not p or not Path(p).exists():
            raise ValueError("Choose the annotation file (or the annotation folder).")
        self.need(self.images)
        if not self.out.text().strip():
            raise ValueError("Choose the output folder.")
        return IM.plan(p, self.images.text().strip(), self.out.text().strip(), self.fmt.currentData(),
                       self.output.currentData(), self.masks.isChecked(), self.link.isChecked(), progress, cancel,
                       spec=self._mask_spec() if self.mask_box.isVisible() else
                       ({"attr": self.via_attr.currentText()} if self.via_attr.isVisible() else self._spec))

    def run(self, plan, cancel, progress):
        return IM.run(plan, progress, cancel)

    def result_lines(self, st):
        return [f"{st['done']:,} image(s) with {st['shapes']:,} shape(s) written · {st['linked']:,} linked, "
                f"{st['copied']:,} copied" + (f" · {st['skipped']:,} shape(s) not representable (points / lines in "
                                              "YOLO)" if st["skipped"] else "")]

    def out_folder(self, plan, st):
        return plan.opts["out"]

    def dataset_folder(self, plan, st):
        return plan.opts["out"]


class ExifPage(Page):
    TITLE, ICON = "Fix EXIF rotation", "redo"
    HELP = ("Phone photos often store the pixels sideways plus an EXIF “rotate me” tag — labelme, viewers and "
            "Ultralytics apply it, OpenCV-based tools do not, so boxes land in different places. This bakes the "
            "rotation into the pixels (tag reset; other EXIF data and the colour profile kept) and brings LabelMe / "
            "YOLO labels into the same upright frame. In place — every file is kept first, so Undo restores it.")

    def build(self):
        self.root = self.folder("Folder:", "exif_root")
        self.policy = QComboBox()
        for t, v in (("auto — decided per file from the label size (recommended)", "auto"),
                     ("the labels were drawn on the upright view (labelme, Ultralytics)", "upright"),
                     ("the labels were drawn on the raw, un-rotated pixels", "raw")):
            self.policy.addItem(t, v)
        self.form.addRow("Labels:", self.policy)

    def make_plan(self, cancel, progress):
        self.need(self.root)
        return X.plan_folder(self.root.text().strip(), self.policy.currentData(), progress, cancel)

    def run(self, plan, cancel, progress):
        return X.run(plan, Path(self.root.text().strip()) / C.LOG_DIR, progress, cancel)

    def result_lines(self, st):
        return [f"{st['done']:,} image(s) made upright · {st['labels']:,} label file(s) rotated with them"]

    def out_folder(self, plan, st):
        return self.root.text().strip()

    def dataset_folder(self, plan, st):
        return self.root.text().strip()


class PickPage(Page):
    TITLE, ICON = "Pick images to label", "review-next"
    HELP = ("From a folder of unlabelled images, pick the ones worth labelling first: the most varied set (each pick "
            "is the image least like everything picked so far — near-identical frames are not picked twice), "
            "optionally unlike the images of the open dataset and where its model is unsure. The picks are copied "
            "(or moved) to a folder in rank order (picked.csv): label from the top.")
    RUN = "Copy / Move"

    def build(self):
        self.src = self.folder("Unlabelled pool:", "pk_src")
        self.out = self.folder("Output folder:", "pk_out", "Where the picked images go")
        self.n = QSpinBox()
        self.n.setRange(1, 1_000_000)
        self.n.setValue(int(self.st.value("prep/pk_n", 200)))
        self.n.valueChanged.connect(lambda v: self.st.setValue("prep/pk_n", v))
        self.form.addRow("Pick:", self.n)
        self.feat = QComboBox()
        self.feat.addItem("Built-in image features (fast, no model)", "builtin")
        self.feat.addItem("Visual Similarity model (DINOv3 / SigLIP2 / CLIP)", "model")
        self.feat.setToolTip("The model is the one last used in Models ▸ Visual Similarity, run in the Python "
                             "environment of Models & Environment")
        self.feat.setCurrentIndex(max(0, self.feat.findData(self.st.value("prep/pk_feat", "builtin"))))
        self.feat.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/pk_feat", self.feat.currentData()))
        self.form.addRow("Compare by:", self.feat)
        self.mode = QComboBox()
        self.mode.addItem("Copy (the pool stays as it is)", "copy")
        self.mode.addItem("Move (out of the pool)", "move")
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.st.value("prep/pk_mode", "copy"))))
        self.mode.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/pk_mode", self.mode.currentData()))
        self.form.addRow("Then:", self.mode)
        self.skip = self.cb("Skip images that already have a label file", "pk_skip", True)
        self.unlike = self.cb("Prefer images unlike the open dataset's (up to 3,000 of its images are compared)",
                              "pk_unlike", True)
        self.unsure = self.cb("Prefer images the open dataset's model is unsure about (Models & Environment)",
                              "pk_unsure", False)
        self.stubs = self.cb("Make an empty label file for each pick — label them where they are (Image QC: draw, "
                             "or accept the model's suggestions)", "pk_stubs", False,
                             "LabelMe JSONs next to the images, or YOLO txt files in the matching labels folder. "
                             "Picks inside the open dataset's folder join it at once, flagged “to label”.")
        self.sort_cls = self.cb("Classifier: sort the picks into class folders by the model's best guess (then "
                                "confirm or move them in the Class Gallery)", "pk_sort_cls", True,
                                "With “model is unsure” and a classification model: each pick lands in "
                                "<output>/<guessed class>/ — open the output folder as a class-folder dataset")
        for c in (self.skip, self.unlike, self.unsure, self.stubs, self.sort_cls):
            self.form.addRow(c)
        self.model_path = ""              # Improve Model: judge with this model instead of the first enabled one

    # ---- worker thread
    def make_plan(self, cancel, progress):
        import numpy as np
        self.need(self.src)
        if not self.out.text().strip():
            raise ValueError("Choose the output folder.")
        app = self.dlg.app
        session = getattr(app, "session", None)
        src = self.src.text().strip()
        progress(0, 0, "Listing the pool")
        paths, n_lab = PK.pool(src, self.skip.isChecked(), cancel)
        if not paths:
            raise ValueError("No unlabelled images in that folder.")
        use_model = self.feat.currentData() == "model"
        embed = self._embedder(app) if use_model else None
        X, ok = embed(paths, cancel, progress) if embed else PK.feature_matrix(paths, progress, cancel)
        seeds = None
        if self.unlike.isChecked() and session is not None:
            import random
            ds = [str(s.image_path) for s in session.samples if s.image_path is not None]
            ds = random.Random(1).sample(ds, 3000) if len(ds) > 3000 else ds
            if ds:
                S, sok = embed(ds, cancel, progress, "Reading the dataset's images") if embed else \
                    PK.feature_matrix(ds, progress, cancel, "Reading the dataset's images")
                seeds = S[sok]
        unc = None
        self._guess = None
        if self.unsure.isChecked():
            if session is None:
                raise ValueError("Open the dataset whose model should judge the pool (or untick “model is unsure”).")
            unc = self._uncertainty(app, session, paths, cancel, progress)
        weight = None if unc is None else 1.0 + np.clip(unc / max(1e-9, float(np.percentile(unc, 95))), 0, 1)
        picked, novelty, mind = PK.k_center(X, self.n.value(), seeds, weight, ok, cancel=cancel, progress=progress)
        guess = self._guess if self.sort_cls.isChecked() else None
        P = PK.plan(src, self.out.text().strip(), [paths[i] for i in picked],
                    [(novelty[j], None if unc is None else float(unc[i])) for j, i in enumerate(picked)],
                    copy=self.mode.currentData() == "copy",
                    folders=[guess[i] for i in picked] if guess is not None else None)
        nov = [v for v in novelty if v == v]
        near = 0.5 * float(np.median(nov)) if nov else 0.0      # "close": half the typical gap between picks
        close = max(0, int(((mind < near) & ok).sum()) - len(picked))
        P.rows = [("Images in the pool", len(paths) + n_lab), ("    already labelled (skipped)", n_lab),
                  ("    unreadable", int((~ok).sum())), ("Picked", len(picked)),
                  ("Compared by", self.feat.currentText()),
                  ("Unlike the open dataset", "yes" if seeds is not None else "no"),
                  ("Model uncertainty", "yes" if unc is not None else "no"),
                  ("Sorted into class folders by the classifier's guess", "yes" if guess is not None else "no"),
                  ("Images left that look like a pick (close look-alikes)", close)]
        P.details["Picked, in rank order (label from the top)"] = [
            f"{j + 1:>4}. {os.path.relpath(paths[i], src)}" + ("" if novelty[j] != novelty[j] else
                                                               f"   novelty {novelty[j]:.3f}")
            + ("" if unc is None else f"   uncertainty {float(unc[i]):.2f}") for j, i in enumerate(picked)]
        return P

    def _embedder(self, app):
        import tempfile
        import numpy as np
        path = app.settings.value("embed/last", "") or ""
        py = getattr(getattr(app, "inference", None), "python", "")
        if not path or not Path(path).is_dir():
            raise ValueError("No Visual Similarity model yet — set one up in Models ▸ Visual Similarity, or "
                             "compare by built-in features.")
        if not py:
            raise ValueError("Choose the Python environment in Models & Environment first.")
        svc = app.embedder
        batch = int(app.settings.value("embed/batch", 32))

        def embed(paths, cancel, progress, label="Embedding images"):
            progress(0, 0, "Starting the model")
            svc.probe(py)
            svc.ensure_loaded(py, path, "auto", False)
            tmp = Path(tempfile.mkdtemp(prefix="aioqc_pick_"))
            vecs = []
            failed = set()
            try:
                for a in range(0, len(paths), 256):
                    if cancel.is_set():
                        raise C.Cancelled()
                    part = paths[a:a + 256]
                    res = svc.embed([{"image": p, "boxes": [None]} for p in part], tmp / f"{a}.npy", batch, 0.0,
                                    cancel=cancel)
                    for n, _k, _e in res.get("failed") or []:
                        failed.add(a + int(n))
                    vecs.append(np.load(tmp / f"{a}.npy").astype(np.float32))
                    progress(min(a + 256, len(paths)), len(paths), label)
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            X = np.concatenate(vecs) if vecs else np.zeros((0, 1), np.float32)
            ok = np.ones(len(X), bool)
            for i in failed:
                if i < len(ok):
                    ok[i] = False
            return X, ok
        return embed

    def _uncertainty(self, app, session, paths, cancel, progress):
        import numpy as np
        from AIO_QC.core.state import load_models_config
        from .model_setup_dialog import server_specs
        svc = app.inference
        models = [m for m in load_models_config(session.root).get("models", []) if m.get("enabled", True)]
        if self.model_path:                 # chosen by Improve Model (the model being improved)
            from AIO_QC.core.model_registry import _norm
            want = [m for m in load_models_config(session.root).get("models", [])
                    if _norm(m.get("path", "")) == _norm(self.model_path)]
            models = want or models
        if not models or not svc.python:
            raise ValueError("Add a model and its environment in Models & Environment first (or untick “model is "
                             "unsure”).")
        progress(0, 0, "Starting the model")
        svc.ensure_started()
        specs = server_specs(models, svc.opts, svc.info)[:1]
        if not specs:
            raise ValueError("The model file of the open dataset was not found.")
        info = svc.ensure_loaded(specs) or {}
        mid = specs[0]["id"]
        names = {int(k): v for k, v in ((info.get(mid) or {}).get("names") or {}).items()}
        cls_model = (info.get(mid) or {}).get("task") == "classify" or bool(getattr(session, "is_classification",
                                                                                    False))
        unc = np.zeros(len(paths), np.float32)
        guess = [None] * len(paths)
        for a in range(0, len(paths), 64):
            if cancel.is_set():
                raise C.Cancelled()
            part = paths[a:a + 64]
            res = svc.predict(part, [mid], conf=0.0 if cls_model else 0.1, cancel=cancel)
            for j, p in enumerate(part):
                boxes = ((res.get(str(p)) or {}).get("by_model") or {}).get(mid) or []
                from AIO_QC.tools.frame_extractor.core.detector import is_cls_rows
                if cls_model and boxes and is_cls_rows(np.asarray(boxes, np.float32).reshape(-1, 6)):
                    u, best = PK.cls_uncertainty(boxes)            # a classifier: a close call between two classes
                    unc[a + j] = u
                    guess[a + j] = names.get(best, str(best)) if best is not None else None
                    continue
                # each prediction between "surely not" and "surely yes" adds up to 1 (most at conf 0.5)
                unc[a + j] = sum(1.0 - abs(2.0 * float(b[4]) - 1.0) for b in boxes if 0.1 <= float(b[4]) <= 0.9)
            progress(min(a + 64, len(paths)), len(paths), "Asking the model")
        if any(g is not None for g in guess):
            self._guess = guess
        return unc

    def run(self, plan, cancel, progress):
        st = PK.run(plan, progress, cancel)
        session = getattr(self.dlg.app, "session", None)
        if self.stubs.isChecked() and not getattr(session, "is_classification", False):
            from AIO_QC.core.improve import label_stubs
            dsts = [a[2] for a in plan.actions if os.path.isfile(a[2])]
            yolo = any("/images/" in d.replace("\\", "/") for d in dsts)
            st["stubs"] = label_stubs(dsts, "yolo" if yolo else "labelme")
        return st

    def result_lines(self, st):
        out = [f"{st['done']:,} image(s) {'copied' if self.mode.currentData() == 'copy' else 'moved'} — ranked "
               "list in picked.csv"]
        if st.get("stubs") is not None:
            out.append(f"{len(st['stubs']):,} empty label file(s) made" +
                       (f" — {st['joined']:,} joined the open dataset, flagged “to label”" if st.get("joined") else ""))
        return out

    def after_run(self, plan, st):
        """UI thread: picks with label files inside the open dataset's folder join it at once, flagged “to label”."""
        session = getattr(self.dlg.app, "session", None)
        stubs = st.get("stubs") or []
        if session is None or not stubs:
            return
        root = os.path.abspath(str(session.root))
        mine = [p for p in stubs if os.path.abspath(p).startswith(root + os.sep)]
        new = []
        for p in mine:
            ann = p if p.endswith(".json") else next((str(Path(p).parent.parent / "images" / (Path(p).stem + x))
                                                     for x in (".jpg", ".png", ".jpeg", ".bmp", ".webp")
                                                     if (Path(p).parent.parent / "images" / (Path(p).stem + x)).exists()),
                                                    p)
            smp, _structural = session.adapter.sample_for_path(ann)
            if smp is not None:
                new.append(smp)
        if new:
            keys = session.add_samples(new)
            if keys:
                from AIO_QC.core.state import REVIEW_FLAG
                session.set_review(keys, REVIEW_FLAG, "to label — picked by Pick images to label")
            st["joined"] = len(keys)

    def out_folder(self, plan, st):
        return plan.opts["out"]


def _sources_box(page, key, placeholder, title):
    """A list of folders, one per line, with Add…; remembered."""
    edit = QPlainTextEdit(page.st.value(f"prep/{key}", "") or "")
    edit.setPlaceholderText(placeholder)
    edit.setFixedHeight(64)
    edit.textChanged.connect(lambda: page.st.setValue(f"prep/{key}", edit.toPlainText()))

    def add():
        d = QFileDialog.getExistingDirectory(page, title)
        if d:
            t = edit.toPlainText().rstrip()
            edit.setPlainText((t + "\n" if t else "") + d)
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    lay.addWidget(newButton("Add…", icon="open", slot=add), 0, Qt.AlignmentFlag.AlignTop)
    return w, edit


def _folders(edit) -> list:
    return [x.strip().strip('"') for x in edit.toPlainText().splitlines() if x.strip()]


def _rename_map(text: str) -> dict:
    """"truck_text = trailer_text, Car=car" -> {"truck_text": "trailer_text", "Car": "car"}"""
    out = {}
    for part in re.split(r"[,;\n]+", text or ""):
        if "=" in part:
            a, b = part.split("=", 1)
            if a.strip() and b.strip():
                out[a.strip()] = b.strip()
    return out


def _dataset_kind(root: str) -> str:
    """A quick look (top folders only) for the line under the dataset folder."""
    p = Path(root)
    if not root or not p.is_dir():
        return ""
    names = ("train", "val", "valid", "test")
    sp = [d.name for d in p.iterdir() if d.is_dir() and d.name.lower() in names]
    yolo = (p / "images").is_dir() or (p / "labels").is_dir() or any((p / d / "images").is_dir() for d in sp)
    if not sp and (p / "images").is_dir():                         # images/train, images/val (the usual YOLO layout)
        sp = [d.name for d in (p / "images").iterdir() if d.is_dir() and d.name.lower() in names]
    cls = next((n for n in ("data.yaml", "dataset.yaml", "classes.txt") if (p / n).is_file()), "")
    return ("YOLO" if yolo else "LabelMe") + (f" · splits: {', '.join(sorted(sp))}" if sp else " · no splits") + \
        (f" · class list: {cls}" if yolo and cls else "")


class _AddBase(Page):
    """What Add new data and Add classes by image share: the dataset, the folders, the class rename map."""

    def combo(self, label, key, items, default, tip=""):
        c = QComboBox()
        for val, text in items:
            c.addItem(text, val)
        c.setCurrentIndex(max(0, c.findData(self.st.value(f"prep/{key}", default))))
        c.currentIndexChanged.connect(lambda _i: self.st.setValue(f"prep/{key}", c.currentData()))
        c.setToolTip(tip)
        self.form.addRow(label, c)
        return c

    def dataset_row(self, key):
        self.dataset = self.folder("Dataset:", key, "The existing dataset (LabelMe folder, or YOLO root)")
        self.kind = QLabel("")
        apply_style(self.kind, "color:@muted;")
        self.form.addRow("", self.kind)
        self.dataset.textChanged.connect(lambda t: self.kind.setText(_dataset_kind(t.strip())))
        self.kind.setText(_dataset_kind(self.dataset.text().strip()))

    def rename_row(self, key):
        self.rename = QLineEdit(self.st.value(f"prep/{key}", "") or "")
        self.rename.setPlaceholderText("new label = dataset class, …   e.g.  truck_text = trailer_text, Car = car")
        self.rename.textChanged.connect(lambda t: self.st.setValue(f"prep/{key}", t))
        self.rename.setToolTip("Labels the new data calls differently. Letter case and extra spaces never matter "
                               "(Car = car = “car ” anyway).")
        self.form.addRow("Same class, other name:", self.rename)

    def result_lines(self, st):
        d = st["done"]
        out = [f"{d.get('added', 0):,} images added", f"{d.get('updated', 0):,} annotations changed"]
        if d.get("classes"):
            out.append(f"{d['classes']} classes added to the class list")
        return out + [f"report: {Path(st['report']).name}"]

    def run(self, plan, cancel, progress):
        from AIO_QC.core.prep import add_data as AD
        return AD.run(plan, progress, cancel)

    def out_folder(self, plan, st):
        return plan.target.root

    def dataset_folder(self, plan, st):
        return plan.target.root


class AddDataPage(_AddBase):
    TITLE, ICON = "Add new data", "data-add-step"
    HELP = ("New image + annotation pairs into an existing dataset — LabelMe, or YOLO with or without train / val / "
            "test — matched by the IMAGE, not the file name: an image the dataset already has is recognised however "
            "it was renamed (and listed when its annotation differs). The new data may be LabelMe or YOLO, nested "
            "any way, several folders; it is converted to the dataset's format, class ids by name. A name the "
            "dataset already uses for a different image gets a new one by the naming scheme below. Nothing is "
            "changed in the new data; Undo removes what the run added and restores the class lists.")

    def build(self):
        self.dataset_row("ad_dataset")
        w, self.sources = _sources_box(self, "ad_sources", "One folder of new data per line (LabelMe or YOLO pairs, "
                                                           "any nesting) — or Add…", "Folder of new data")
        self.form.addRow("New data:", w)
        self.on_match = self.combo("Images it already has:", "ad_on_match", [
            ("skip", "Leave them (list those whose annotation differs)"),
            ("replace", "Replace the dataset's annotation with the new one"),
            ("merge", "Add the classes the dataset's annotation lacks")], "skip",
            "Matched by image content: renamed or moved copies count too")
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.split_mode = QComboBox()
        for val, text in (("train", "All to train"), ("balanced", "Balanced split"),
                          ("source", "As the new data's own train / val / test")):
            self.split_mode.addItem(text, val)
        self.split_mode.setCurrentIndex(max(0, self.split_mode.findData(self.st.value("prep/ad_split", "train"))))
        self.split_mode.currentIndexChanged.connect(lambda _i: (self.st.setValue("prep/ad_split",
                                                                                 self.split_mode.currentData()),
                                                                self._split_ui()))
        self.like = QCheckBox("like the dataset")
        self.like.setChecked(self.st.value("prep/ad_like", True, type=bool))
        self.like.toggled.connect(lambda on: (self.st.setValue("prep/ad_like", on), self._split_ui()))
        self.val = QDoubleSpinBox()
        self.test = QDoubleSpinBox()
        for sp, key, dflt, lab in ((self.val, "ad_val", 10.0, "val"), (self.test, "ad_test", 0.0, "test")):
            sp.setRange(0, 60)
            sp.setSuffix(f" % {lab}")
            sp.setDecimals(0)
            sp.setValue(float(self.st.value(f"prep/{key}", dflt)))
            sp.valueChanged.connect(lambda v, k=key: self.st.setValue(f"prep/{k}", v))
        for x in (self.split_mode, self.like, self.val, self.test):
            h.addWidget(x)
        h.addStretch(1)
        row.setToolTip("A dataset with train / val / test: where the new images go. Balanced: like the dataset's "
                       "own shares (or yours) — frames of one video stay together, classes seen in few images stay "
                       "in train. A dataset without splits takes them as they are.")
        self.form.addRow("New images go to:", row)
        self.new_classes = self.combo("Classes the dataset lacks:", "ad_new_classes", [
            ("add", "Add them to its class list (YOLO)"), ("drop", "Drop those objects"),
            ("skip", "Leave those images out")], "add", "YOLO datasets: their class list (classes.txt, data.yaml) "
                                                        "is extended in step; LabelMe keeps every label")
        self.rename_row("ad_rename")
        self.unlabelled = self.combo("Images without annotation:", "ad_unlabelled", [
            ("skip", "Leave them out"), ("add", "Add them (background — no objects)")], "skip")
        prow = QWidget()
        ph = QHBoxLayout(prow)
        ph.setContentsMargins(0, 0, 0, 0)
        self.placement = QComboBox()
        self.placement.addItem("Keep the new data's sub-folders", "mirror")
        self.placement.addItem("All in one folder", "flat")
        self.placement.setCurrentIndex(max(0, self.placement.findData(self.st.value("prep/ad_place", "mirror"))))
        self.placement.currentIndexChanged.connect(lambda _i: self.st.setValue("prep/ad_place",
                                                                               self.placement.currentData()))
        self.subfolder = QLineEdit(self.st.value("prep/ad_sub", "") or "")
        self.subfolder.setPlaceholderText("under this folder (empty: the dataset's own)")
        self.subfolder.textChanged.connect(lambda t: self.st.setValue("prep/ad_sub", t))
        ph.addWidget(self.placement)
        ph.addWidget(self.subfolder, 1)
        self.form.addRow("LabelMe dataset:", prow)
        self.link = self.combo("Files:", "ad_link", [("copy", "Copies"),
                                                     ("hardlink", "Hard links (no extra space; copies where not "
                                                                  "possible)")], "copy")
        self.check = self.cb("Check that every new image opens (a full decode)", "ad_check", True)
        self.form.addRow(self.check)
        self.naming = _naming_box(self, "add_data")
        if not self.st.value("naming/last/prep_add_data", ""):
            from AIO_QC.core import naming as N
            self.naming.set_scheme(N.Scheme(preset="suffix"))
            self._naming = self.naming.scheme().to_dict()
        self.naming.setToolTip("A new image whose name the dataset already uses for a DIFFERENT image (or two new "
                               "images with one name): how it is renamed. Only those are renamed, unless you choose "
                               "otherwise; every name is checked against the dataset's.")
        self.form.addRow("Names taken:", self.naming)
        self._split_ui()

    def _split_ui(self):
        bal = self.split_mode.currentData() == "balanced"
        self.like.setVisible(bal)
        for sp in (self.val, self.test):
            sp.setVisible(bal)
            sp.setEnabled(not self.like.isChecked())

    def make_plan(self, cancel, progress):
        from AIO_QC.core.prep import add_data as AD
        self.need(self.dataset)
        srcs = _folders(self.sources)
        bal = self.split_mode.currentData() == "balanced" and not self.like.isChecked()
        o = AD.Options(self.dataset.text().strip(), srcs, add_new=True, on_match=self.on_match.currentData(),
                       split_mode=self.split_mode.currentData(), val=self.val.value() / 100 if bal else None,
                       test=self.test.value() / 100 if bal else None, new_classes=self.new_classes.currentData(),
                       rename=_rename_map(self.rename.text()), unlabelled=self.unlabelled.currentData(),
                       placement=self.placement.currentData(), subfolder=self.subfolder.text().strip(),
                       link=self.link.currentData(), check_images=self.check.isChecked(), naming=dict(self._naming))
        return AD.plan(o, progress, cancel)


class AddClassesPage(_AddBase):
    TITLE, ICON = "Add classes by image", "tag"
    HELP = ("Bring annotations of more classes to the images a dataset already has — matched by the IMAGE, not the "
            "file name. For example a LabelMe master with every class and a YOLO dataset converted with a few: the "
            "YOLO labels get the other classes, its class list extended. Any direction (LabelMe ↔ YOLO, detection ↔ "
            "segmentation: boxes and polygons converted). The dataset images the source does not have are listed — "
            "they would lack the added classes. Undo restores every annotation and class list.")

    def build(self):
        self.dataset_row("ac_dataset")
        w, self.sources = _sources_box(self, "ac_sources", "The folder(s) with the same images and more classes "
                                                           "(LabelMe or YOLO)", "Folder with more classes")
        self.form.addRow("Classes from:", w)
        self.classes = QLineEdit(self.st.value("prep/ac_classes", "") or "")
        self.classes.setPlaceholderText("empty: every class the dataset does not have   ·   or: plate, person")
        self.classes.textChanged.connect(lambda t: self.st.setValue("prep/ac_classes", t))
        self.form.addRow("Classes:", self.classes)
        self.conflict = self.combo("A class both have in an image:", "ac_conflict", [
            ("keep", "Keep the dataset's objects"),
            ("union", "Add the source's objects that don't overlap them"),
            ("replace", "Replace them with the source's")], "keep",
            "Chosen classes the dataset already has in an image (e.g. some images only partly labelled)")
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.1, 0.95)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(float(self.st.value("prep/ac_iou", 0.6)))
        self.iou.valueChanged.connect(lambda v: self.st.setValue("prep/ac_iou", v))
        self.iou.setToolTip("Overlap (IoU) from which a source object counts as already there")
        self.form.addRow("Same object from overlap:", self.iou)
        self.rename_row("ac_rename")
        self.add_new = self.cb("Also add the source's images the dataset does not have (to train, names kept "
                               "unique)", "ac_add_new", False)
        self.form.addRow(self.add_new)

    def make_plan(self, cancel, progress):
        from AIO_QC.core.prep import add_data as AD
        self.need(self.dataset)
        classes = [c.strip() for c in re.split(r"[,;]+", self.classes.text()) if c.strip()] or None
        o = AD.Options(self.dataset.text().strip(), _folders(self.sources), add_new=self.add_new.isChecked(),
                       on_match="merge", merge_classes=classes, conflict=self.conflict.currentData(),
                       iou=self.iou.value(), rename=_rename_map(self.rename.text()))
        return AD.plan(o, progress, cancel)


class UndoPage(Page):
    TITLE, ICON = "Undo a run", "undo"
    HELP = ("Every Preparation run is logged in <folder>/_aioqc_prep/. Choose the folder the run wrote to (the flat / "
            "output folder, or the folder you worked in), pick the run, Undo: moved files go back, files it created "
            "are removed (only if unchanged since), rewritten JSONs get their original bytes back. Nothing is "
            "overwritten.")
    RUN = "Undo"

    def build(self):
        self.root = self.folder("Folder:", "undo_root")
        self.runs = QListWidget()
        self.runs.setMinimumHeight(120)
        self.root.textChanged.connect(lambda _t: self._fill())
        self.form.addRow("Runs:", self.runs)
        self._fill()

    def _fill(self):
        self.runs.clear()
        p = self.root.text().strip()
        for j in C.list_runs(p) if p and Path(p).is_dir() else []:
            if j.name.endswith(".undone.jsonl"):
                continue
            it = QListWidgetItem(j.stem)
            it.setData(Qt.ItemDataRole.UserRole, str(j))
            self.runs.addItem(it)
        if self.runs.count():
            self.runs.setCurrentRow(0)

    def make_plan(self, cancel, progress):
        it = self.runs.currentItem()
        if it is None:
            raise ValueError("No run selected (choose the folder the run wrote to).")
        j = it.data(Qt.ItemDataRole.UserRole)
        n = sum(1 for _ in open(j, encoding="utf-8")) - 1
        P = T.Plan("undo", opts={"jsonl": j})
        P.rows = [("Run", Path(j).stem), ("Recorded operations", n)]
        return P

    def run(self, plan, cancel, progress):
        return C.undo_run(plan.opts["jsonl"], progress)

    def result_lines(self, st):
        return [f"{st['restored']:,} restored · {st['removed']:,} removed · {len(st['skipped'])} left alone"]

    def after(self):
        self._fill()


PAGES = [UnzipPage, ImportPage, FlattenPage, CombineYoloPage, AddDataPage, AddClassesPage, OrphansPage, IntegrityPage,
         DupesPage, ExifPage, ResizePage, PickPage, UndoPage]


# ====================================================================================================== dialog
def _provenance_record(pg, P, st):
    """The folder a Preparation run wrote or changed gets a provenance record (in the background: the output's
    fingerprint lists every file)."""
    from AIO_QC.core import provenance as PV
    if not PV.enabled():
        return
    try:
        folder = pg.dataset_folder(P, st) or pg.out_folder(P, st)
    except Exception as exc:                             # noqa: BLE001 — a record never breaks a run
        _ignored(exc)
        return
    if not folder or not Path(str(folder)).is_dir():
        return
    folder = Path(str(folder))
    opts = getattr(P, "opts", None)
    opts = vars(opts) if opts is not None and not isinstance(opts, dict) and hasattr(opts, "__dict__") else (opts or {})
    inputs = []
    for v in opts.values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, (str, Path)) and len(str(x)) > 1 and Path(str(x)).is_dir() and Path(str(x)) != folder:
                inputs.append(str(x))
    counts = {k: v for k, v in st.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    title = pg.TITLE
    o = PV.make_origin("rule", f"Preparation — {title}")
    import threading
    threading.Thread(target=PV.record, args=(folder, "prepare", f"Preparation — {title}"),
                     kwargs={"inputs": list(dict.fromkeys(inputs))[:6], "settings": opts, "counts": counts, "o": o,
                             "input_fingerprints": False},
                     daemon=True, name="provenance").start()


class DataPrepDialog(QDialog):
    TITLE = "Preparation"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-prep"))
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        self.plan = None
        self.job = None
        self.nav = QListWidget()
        self.nav.setFixedWidth(200)
        self.stack = QStackedWidget()
        self.pages = []
        for cls in PAGES:
            pg = cls(self)
            self.pages.append(pg)
            self.stack.addWidget(pg)
            self.nav.addItem(QListWidgetItem(newIcon(cls.ICON), cls.TITLE))
        self.nav.currentRowChanged.connect(self._switch)
        self.summary = QTreeWidget()
        self.summary.setHeaderLabels(["", "Count"])
        self.summary.setRootIsDecorated(False)
        self.summary.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.details = QTreeWidget()
        self.details.setHeaderLabels(["Files concerned"])
        self.status = QLabel("Choose the folders, then Preview — nothing changes until you Run.")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.preview_btn = newButton("Preview", icon="zoom", slot=self._preview,
                                     tip="Work out what would happen — nothing changes")
        self.run_btn = newButton("Run", icon="review-ok", slot=self._run)
        self.run_btn.setEnabled(False)
        self.stop_btn = newButton("Stop", slot=self._stop)
        self.stop_btn.setVisible(False)
        self.open_btn = newButton("Open Folder", icon="open", slot=self._open_folder)
        self.ds_btn = newButton("Open as Dataset", icon="open", slot=self._open_dataset)
        self.log_btn = newButton("Log", icon="logs", slot=self._open_log)
        self.undo_btn = newButton("Undo This Run", icon="undo", slot=self._undo_last)
        for b in (self.open_btn, self.ds_btn, self.log_btn, self.undo_btn):
            b.setVisible(False)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.addWidget(self.stack)
        res = QSplitter(Qt.Orientation.Horizontal)
        res.addWidget(self.summary)
        res.addWidget(self.details)
        res.setSizes([420, 520])
        rl.addWidget(res, 1)
        rl.addWidget(self.status)
        foot = QHBoxLayout()
        foot.addWidget(self.progress, 1)
        for b in (self.open_btn, self.ds_btn, self.log_btn, self.undo_btn):
            foot.addWidget(b)
        foot.addStretch(1)
        foot.addWidget(self.preview_btn)
        foot.addWidget(self.run_btn)
        foot.addWidget(self.stop_btn)
        foot.addWidget(newButton("Close", slot=self.close))
        rl.addLayout(foot)
        lay = QHBoxLayout(self)
        lay.addWidget(self.nav)
        lay.addWidget(right, 1)
        self.last = None
        want = app.settings.value("prep/page_title", "") or ""
        row = next((n for n, c in enumerate(PAGES) if c.TITLE == want), None)
        if row is None:
            try:
                row = int(app.settings.value("prep/page", 1))
            except (TypeError, ValueError):
                row = 1
        self.nav.setCurrentRow(min(max(row, 0), len(PAGES) - 1))

    def page(self) -> Page:
        return self.pages[self.stack.currentIndex()]

    def _switch(self, i):
        if self.job is not None:
            self.nav.blockSignals(True)
            self.nav.setCurrentRow(self.stack.currentIndex())
            self.nav.blockSignals(False)
            return
        anim.crossfade(self.stack, 160) if self.isVisible() else None
        self.stack.setCurrentIndex(i)
        self.app.settings.setValue("prep/page", i)
        self.app.settings.setValue("prep/page_title", type(self.page()).TITLE)
        self.plan = None
        self.summary.clear()
        self.details.clear()
        self.run_btn.setText(self.page().RUN)
        self.run_btn.setEnabled(isinstance(self.page(), UndoPage))
        self.preview_btn.setVisible(not isinstance(self.page(), UndoPage))
        for b in (self.open_btn, self.ds_btn, self.log_btn, self.undo_btn):
            b.setVisible(False)
        self.status.setText("Choose the folders, then Preview — nothing changes until you Run."
                            if not isinstance(self.page(), UndoPage) else "Pick a run, then Undo.")

    # ------------------------------------------------------------------ background
    def _start(self, fn, done, label):
        self.job = Job(fn, self)
        from AIO_QC.widgets import tasks
        self._task = tasks.start(f"Preparation · {self.page().TITLE} · {label.lower()}", cancel=self._stop)
        self.job.progress.connect(self._on_progress)
        self.job.done.connect(lambda r: (self._idle(), done(r)))
        self.job.failed.connect(lambda m: (self._idle(), self.status.setText(f"⚠ {_plain(m)}")))
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.preview_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setVisible(True)
        self.nav.setEnabled(False)
        self.status.setText(label)
        self.job.start()

    def _on_progress(self, d, t, label):
        if getattr(self, "_task", None) is not None:
            self._task.update(d / t if t else None, label)
        if t > 0:
            self.progress.setRange(0, t)
            self.progress.setValue(d)
        else:
            self.progress.setRange(0, 0)
        if label:
            self.status.setText(f"{label}… {f'{d:,} / {t:,}' if t else ''}")

    def _idle(self):
        self.job = None
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        self.progress.setVisible(False)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setVisible(False)
        self.nav.setEnabled(True)
        self.run_btn.setEnabled(self.plan is not None or isinstance(self.page(), UndoPage))

    def _stop(self):
        if self.job is not None:
            self.job.cancel.set()
            self.status.setText("Stopping after the files in progress…")

    # ------------------------------------------------------------------ preview / run
    def _preview(self):
        pg = self.page()
        self.plan = None
        self.summary.clear()
        self.details.clear()
        for b in (self.open_btn, self.ds_btn, self.log_btn, self.undo_btn):
            b.setVisible(False)
        snap = PageSnapshot(pg)
        self._start(lambda cancel, progress: snap.make_plan(cancel, progress), self._planned, "Working out the plan")

    def _planned(self, P):
        self.plan = P
        self._show_plan(P)
        n = len(P.actions)
        self.status.setText(f"Plan ready — {n:,} file operation(s). Nothing has changed yet; Run to carry it out."
                            if n else "Nothing to do.")
        self.run_btn.setEnabled(n > 0)

    def _show_plan(self, P):
        self.summary.clear()
        for label, v in P.rows:
            it = QTreeWidgetItem([str(label), f"{v:,}" if isinstance(v, int) else str(v)])
            it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight)
            if str(label).startswith("    "):
                it.setForeground(0, self.palette().placeholderText())
            self.summary.addTopLevelItem(it)
        self.details.clear()
        for sec, lines in P.details.items():
            if not lines:
                continue
            top = QTreeWidgetItem([f"{sec} ({len(lines):,})"])
            for t in lines[:2000]:
                top.addChild(QTreeWidgetItem([t]))
            if len(lines) > 2000:
                top.addChild(QTreeWidgetItem([f"… {len(lines) - 2000:,} more (all of them are in the run's log)"]))
            self.details.addTopLevelItem(top)
            top.setExpanded("DIFFERENT" in sec or self.details.topLevelItemCount() == 1)

    def _run(self):
        pg = self.page()
        if isinstance(pg, UndoPage):
            try:
                self.plan = pg.make_plan(None, None)
            except ValueError as exc:
                self.status.setText(f"⚠ {exc}")
                return
            if QMessageBox.question(self, self.TITLE, f"Undo {self.plan.rows[0][1]}?") != QMessageBox.StandardButton.Yes:
                return
        elif self.plan is None:
            return
        else:
            n = len(self.plan.actions)
            if QMessageBox.question(self, self.TITLE, f"Carry out {n:,} file operation(s)?\n\nEverything is logged and "
                                    "can be undone (Undo This Run / the Undo page).") != QMessageBox.StandardButton.Yes:
                return
        P = self.plan
        snap = PageSnapshot(pg)
        self._start(lambda cancel, progress: snap.run(P, cancel, progress), lambda st: self._ran(pg, P, st),
                    "Running")

    def _ran(self, pg, P, st):
        after = getattr(pg, "after_run", None)
        if after is not None:
            try:
                after(P, st)
            except Exception as exc:                        # the run itself succeeded: report, carry on
                st.setdefault("errors", []).append(f"after the run: {exc}")
        self.last = (pg, P, st)
        self.plan = None
        self.run_btn.setEnabled(isinstance(pg, UndoPage))
        lines = pg.result_lines(st)
        errs = st.get("errors") or []
        if errs:
            lines.append(f"⚠ {len(errs)} problem(s): {errs[0]}" + (" …" if len(errs) > 1 else ""))
            top = QTreeWidgetItem([f"Problems ({len(errs):,})"])
            for e in errs[:2000]:
                top.addChild(QTreeWidgetItem([e]))
            self.details.insertTopLevelItem(0, top)
            top.setExpanded(True)
        for s in (st.get("skipped") or [])[:2000]:
            self.details.addTopLevelItem(QTreeWidgetItem([f"left alone: {s}"]))
        self.status.setText("✓ " + " · ".join(lines))
        logger.info(f"[data prep] {pg.TITLE}: {' · '.join(lines)}")
        self.open_btn.setVisible(pg.out_folder(P, st) is not None)
        self.ds_btn.setVisible(pg.dataset_folder(P, st) is not None)
        if not isinstance(pg, UndoPage):
            _provenance_record(pg, P, st)
        self.log_btn.setVisible(bool(st.get("log")))
        self.undo_btn.setVisible(bool(st.get("undo")))
        if isinstance(pg, UndoPage):
            pg.after()

    # ------------------------------------------------------------------ result actions
    def _open_folder(self):
        if self.last:
            pg, P, st = self.last
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(pg.out_folder(P, st))))

    def _open_dataset(self):
        if self.last:
            pg, P, st = self.last
            self.app.open_dataset(Path(pg.dataset_folder(P, st)))

    def _open_log(self):
        if self.last and self.last[2].get("log"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last[2]["log"]))

    def _undo_last(self):
        if not self.last or not self.last[2].get("undo"):
            return
        j = self.last[2]["undo"]
        if QMessageBox.question(self, self.TITLE, "Undo this run? Moved files go back, created files are removed "
                                                  "(only if unchanged).") != QMessageBox.StandardButton.Yes:
            return
        pg = self.page()
        self._start(lambda cancel, progress: C.undo_run(j, progress),
                    lambda st: (self.status.setText(
                        f"↶ Undone: {st['restored']:,} restored · {st['removed']:,} removed · "
                        f"{len(st['skipped'])} left alone" + (f" · ⚠ {len(st['errors'])} error(s)" if st['errors']
                                                               else "")),
                        self.undo_btn.setVisible(False), pg), "Undoing")

    def show_page(self, cls) -> "Page":
        """Switch to a page by its class (e.g. PickPage) and return it."""
        i = next(n for n, pg in enumerate(self.pages) if isinstance(pg, cls))
        self.nav.setCurrentRow(i)
        return self.pages[i]

    def closeEvent(self, e):
        if self.job is not None:
            self.job.cancel.set()
            self.job.wait(8000)
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)
