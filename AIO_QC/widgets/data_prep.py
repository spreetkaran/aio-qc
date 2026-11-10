# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Tools ▸ Data Prep (Ctrl+Alt+P) — get raw data ready before it is opened as a dataset. Works without a dataset open.

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
import threading
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
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


class Job(QThread):
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
                      dedupe=self.dedupe.isChecked(), background_images=self.bg.isChecked())
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
    HELP = ("Annotations from another tool → a NEW folder as LabelMe or YOLO (the sources are not touched). COCO JSON "
            "(boxes, polygons, RLE masks), Pascal VOC XML (a folder), CVAT XML (images and video tracks) and Label "
            "Studio JSON (boxes incl. rotated, polygons, points, ellipses; a text area on a region becomes its "
            "description). Images are matched by path, then by name; they are hard-linked (no extra space) or copied.")

    def build(self):
        self.src = QLineEdit(self.st.value("prep/im_src", "") or "")
        self.src.setPlaceholderText("COCO / Label Studio .json, CVAT .xml, or a folder of VOC .xml")
        self.src.textChanged.connect(lambda t: (self.st.setValue("prep/im_src", t), self._guess()))
        bf = newButton("File…", slot=self._pick_file)
        bd = newButton("Folder…", slot=lambda: _pick(self, self.src, "Folder of Pascal VOC .xml files"))
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
        self._guess()

    def _pick_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Annotation file", self.src.text(),
                                           "Annotations (*.json *.xml);;All files (*)")
        if p:
            self.src.setText(p)

    def _guess(self):
        p = self.src.text().strip()
        try:
            f = IM.detect(p) if p and Path(p).exists() else None
        except Exception:
            f = None
        self.detected.setText(f"detected: {IM.FORMATS[f]}" if f else ("" if not p else "format not recognised"))

    def make_plan(self, cancel, progress):
        p = self.src.text().strip()
        if not p or not Path(p).exists():
            raise ValueError("Choose the annotation file (or the folder of VOC .xml files).")
        self.need(self.images)
        if not self.out.text().strip():
            raise ValueError("Choose the output folder.")
        return IM.plan(p, self.images.text().strip(), self.out.text().strip(), self.fmt.currentData(),
                       self.output.currentData(), self.masks.isChecked(), self.link.isChecked(), progress, cancel)

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
        for c in (self.skip, self.unlike, self.unsure):
            self.form.addRow(c)

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
        if self.unsure.isChecked():
            if session is None:
                raise ValueError("Open the dataset whose model should judge the pool (or untick “model is unsure”).")
            unc = self._uncertainty(app, session, paths, cancel, progress)
        weight = None if unc is None else 1.0 + np.clip(unc / max(1e-9, float(np.percentile(unc, 95))), 0, 1)
        picked, novelty, mind = PK.k_center(X, self.n.value(), seeds, weight, ok, cancel=cancel, progress=progress)
        P = PK.plan(src, self.out.text().strip(), [paths[i] for i in picked],
                    [(novelty[j], None if unc is None else float(unc[i])) for j, i in enumerate(picked)],
                    copy=self.mode.currentData() == "copy")
        nov = [v for v in novelty if v == v]
        near = 0.5 * float(np.median(nov)) if nov else 0.0      # "close": half the typical gap between picks
        close = max(0, int(((mind < near) & ok).sum()) - len(picked))
        P.rows = [("Images in the pool", len(paths) + n_lab), ("    already labelled (skipped)", n_lab),
                  ("    unreadable", int((~ok).sum())), ("Picked", len(picked)),
                  ("Compared by", self.feat.currentText()),
                  ("Unlike the open dataset", "yes" if seeds is not None else "no"),
                  ("Model uncertainty", "yes" if unc is not None else "no"),
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
        if not models or not svc.python:
            raise ValueError("Add a model and its environment in Models & Environment first (or untick “model is "
                             "unsure”).")
        progress(0, 0, "Starting the model")
        svc.ensure_started()
        specs = server_specs(models, svc.opts, svc.info)[:1]
        if not specs:
            raise ValueError("The model file of the open dataset was not found.")
        svc.ensure_loaded(specs)
        mid = specs[0]["id"]
        unc = np.zeros(len(paths), np.float32)
        for a in range(0, len(paths), 64):
            if cancel.is_set():
                raise C.Cancelled()
            part = paths[a:a + 64]
            res = svc.predict(part, [mid], conf=0.1, cancel=cancel)
            for j, p in enumerate(part):
                boxes = ((res.get(str(p)) or {}).get("by_model") or {}).get(mid) or []
                # each prediction between "surely not" and "surely yes" adds up to 1 (most at conf 0.5)
                unc[a + j] = sum(1.0 - abs(2.0 * float(b[4]) - 1.0) for b in boxes if 0.1 <= float(b[4]) <= 0.9)
            progress(min(a + 64, len(paths)), len(paths), "Asking the model")
        return unc

    def run(self, plan, cancel, progress):
        return PK.run(plan, progress, cancel)

    def result_lines(self, st):
        return [f"{st['done']:,} image(s) {'copied' if self.mode.currentData() == 'copy' else 'moved'} — ranked "
                "list in picked.csv"]

    def out_folder(self, plan, st):
        return plan.opts["out"]


class UndoPage(Page):
    TITLE, ICON = "Undo a run", "undo"
    HELP = ("Every Data Prep run is logged in <folder>/_aioqc_prep/. Choose the folder the run wrote to (the flat / "
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


PAGES = [UnzipPage, ImportPage, FlattenPage, OrphansPage, IntegrityPage, DupesPage, ExifPage, ResizePage, PickPage,
         UndoPage]


# ====================================================================================================== dialog
class DataPrepDialog(QDialog):
    TITLE = "Data Prep"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-dataprep"))
        self.resize(1040, 720)
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
        self.nav.setCurrentRow(int(app.settings.value("prep/page", 1)))

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
        self.job.progress.connect(self._on_progress)
        self.job.done.connect(lambda r: (self._idle(), done(r)))
        self.job.failed.connect(lambda m: (self._idle(), self.status.setText(f"⚠ {m}")))
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.preview_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setVisible(True)
        self.nav.setEnabled(False)
        self.status.setText(label)
        self.job.start()

    def _on_progress(self, d, t, label):
        if t > 0:
            self.progress.setRange(0, t)
            self.progress.setValue(d)
        else:
            self.progress.setRange(0, 0)
        if label:
            self.status.setText(f"{label}… {f'{d:,} / {t:,}' if t else ''}")

    def _idle(self):
        self.job = None
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
        self._start(lambda cancel, progress: pg.make_plan(cancel, progress), self._planned, "Working out the plan")

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
        self._start(lambda cancel, progress: pg.run(P, cancel, progress), lambda st: self._ran(pg, P, st),
                    "Running")

    def _ran(self, pg, P, st):
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

    def closeEvent(self, e):
        if self.job is not None:
            self.job.cancel.set()
            self.job.wait(8000)
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)
