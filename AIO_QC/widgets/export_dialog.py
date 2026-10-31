# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
📦 Export & Split — build train / val / [test] (or keep the existing splits), see how they look, export a clean dataset.

Split: val is required (default 10 %), test optional; groups stay together (duplicates, frames of one video, one OCR
identity); rare features stay in train; per-feature caps keep val / test representative (core.split).
Preview: a stacked bar for the whole dataset, one bar per class (share of its images in train / val / test, with the
target marked), and per-feature status. Export: YOLO det / seg, LabelMe, COCO, class folders + reports.
"""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import QRectF, QSize, Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QProgressBar, QRadioButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core import export as EX
from AIO_QC.core.split import (SplitItem, build_groups, classification_features, detection_features, make_report,
                               plan_split, sequence_id)
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style, theme_color
from .inference_service import FuncWorker

SPLIT_COLORS = {"train": QColor("#1E88E5"), "val": QColor("#FB8C00"), "test": QColor("#8E24AA")}
ORDER = ("train", "val", "test")


class SplitBar(QWidget):
    """The whole dataset: one stacked bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts: dict = {}
        self.setMinimumHeight(64)

    def set_counts(self, counts):
        self.counts = dict(counts)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        total = sum(self.counts.values())
        r = QRectF(self.rect()).adjusted(2, 4, -2, -26)
        if not total:
            p.setPen(theme_color("muted"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Press Preview to see the split")
            return
        x = r.left()
        f = p.font()
        f.setBold(True)
        for sp in ORDER:
            n = self.counts.get(sp, 0)
            if not n:
                continue
            w = r.width() * n / total
            seg = QRectF(x, r.top(), w, r.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(SPLIT_COLORS[sp])
            p.drawRoundedRect(seg, 6, 6)
            p.setPen(QColor("white"))
            p.setFont(f)
            txt = f"{sp.capitalize()}  {n:,}  ({100 * n / total:.1f} %)"
            if p.fontMetrics().horizontalAdvance(txt) > w - 8:
                txt = f"{100 * n / total:.0f} %"
            p.drawText(seg, Qt.AlignmentFlag.AlignCenter, txt)
            x += w
        p.setPen(theme_color("muted"))
        p.setFont(self.font())
        p.drawText(QRectF(r.left(), r.bottom() + 4, r.width(), 20), Qt.AlignmentFlag.AlignLeft,
                   f"{total:,} images")


class ClassBars(QWidget):
    """One row per class: the share of its images in train / val / test; the target split marked with a dashed line."""
    ROW = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes: dict = {}
        self.target: dict = {}

    def set_data(self, classes: dict, target: dict):
        self.classes, self.target = classes, target
        self.setMinimumHeight(self.ROW * max(1, len(classes)) + 8)
        self.update()

    def sizeHint(self):
        return QSize(600, self.ROW * max(1, len(self.classes)) + 8)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = p.fontMetrics()
        names = list(self.classes)
        name_w = min(220, max((fm.horizontalAdvance(n) for n in names), default=60) + 12)
        count_w = 170
        bar_l, bar_r = name_w, self.width() - count_w - 8
        for i, (name, v) in enumerate(self.classes.items()):
            y = 4 + i * self.ROW
            tot = sum(v.values()) or 1
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(0, y, name_w - 8, self.ROW - 4), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(name, Qt.TextElideMode.ElideMiddle, name_w - 8))
            x = bar_l
            for sp in ORDER:
                n = v.get(sp, 0)
                if not n:
                    continue
                w = (bar_r - bar_l) * n / tot
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(SPLIT_COLORS[sp])
                p.drawRect(QRectF(x, y + 3, w, self.ROW - 10))
                x += w
            # target boundaries (dashed)
            p.setPen(QPen(self.palette().text().color(), 1.2, Qt.PenStyle.DashLine))
            acc = 0.0
            for sp in ORDER[:-1]:
                acc += self.target.get(sp, 0.0)
                if 0 < acc < 1:
                    xx = bar_l + (bar_r - bar_l) * acc
                    p.drawLine(int(xx), y, int(xx), y + self.ROW - 4)
            p.setPen(theme_color("muted"))
            txt = " / ".join(f"{v.get(sp, 0):,}" for sp in ORDER if sp in self.target or v.get(sp))
            p.drawText(QRectF(bar_r + 8, y, count_w, self.ROW - 4), Qt.AlignmentFlag.AlignVCenter, txt)


class ExportDialog(QDialog):
    TITLE, ICON = "Export & Split", "📦"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-export"))
        self.resize(int(app.width() * 0.75), int(app.height() * 0.85))
        self.plan = None
        self.items = None
        self.worker = None
        self.cancel = threading.Event()
        cls = self.session.is_classification
        has_splits = any(s.split for s in self.session.samples)

        # ---------------------------------------------------------- 1 · output
        out_box = QGroupBox("1 · Output")
        of = QFormLayout(out_box)
        r = QHBoxLayout()
        self.out = QLineEdit(str(self.session.root.parent / f"{self.session.root.name}_export"))
        r.addWidget(self.out, 1)
        r.addWidget(newButton("Browse…", icon="open", slot=self._browse))
        w = QWidget()
        w.setLayout(r)
        of.addRow("Folder:", w)
        self.fmt = QComboBox()
        fmts = ["cls_folders"] + (["labelme"] if self.session.type.value == "cls_labelme" else []) if cls \
            else ["yolo_det", "yolo_seg", "labelme", "coco"]
        for f in fmts:
            self.fmt.addItem(EX.FORMATS[f], f)
        self.mode = QComboBox()
        self.mode.addItem("Hard links (no extra disk space; falls back to copies)", "hardlink")
        self.mode.addItem("Copies", "copy")
        fr = QHBoxLayout()
        fr.addWidget(self.fmt)
        fr.addWidget(self.mode, 1)
        w = QWidget()
        w.setLayout(fr)
        of.addRow("Format:", w)
        ex = QHBoxLayout()
        self.ex_flag = QCheckBox("Leave out flagged files")
        self.ex_flag.setChecked(True)
        self.ex_broken = QCheckBox("Leave out broken images")
        self.ex_broken.setChecked(True)
        self.ex_broken.setToolTip("Corrupt / truncated / undecodable images found by the background check or Image Quality")
        self.ex_empty = QCheckBox("Leave out images without a class" if cls else "Leave out images without labels")
        self.ex_empty.setChecked(cls)
        for c in (self.ex_flag, self.ex_broken, self.ex_empty):
            ex.addWidget(c)
            c.toggled.connect(lambda _on: self._stale())
        ex.addStretch(1)
        w = QWidget()
        w.setLayout(ex)
        of.addRow("Content:", w)
        mapping = getattr(self.session.adapter, "mapping", None)
        if mapping:
            ml = QLabel(f"The current label mapping is applied ({len(mapping)} label(s) renamed / merged).")
            apply_style(ml, "color:@muted;")
            of.addRow("", ml)

        # ---------------------------------------------------------- 2 · split
        sp_box = QGroupBox("2 · Split")
        sf = QFormLayout(sp_box)
        self.keep = QRadioButton("Keep the existing splits" + ("" if has_splits else " (this dataset has none)"))
        self.new = QRadioButton("Build a new split")
        self.keep.setEnabled(has_splits)
        (self.keep if has_splits else self.new).setChecked(True)
        g = QButtonGroup(self)
        g.addButton(self.keep)
        g.addButton(self.new)
        rr = QHBoxLayout()
        rr.addWidget(self.keep)
        rr.addWidget(self.new)
        rr.addStretch(1)
        w = QWidget()
        w.setLayout(rr)
        sf.addRow(w)
        ratio = QHBoxLayout()
        self.val = QSpinBox()
        self.val.setRange(1, 50)
        self.val.setValue(10)
        self.val.setSuffix(" % val")
        self.use_test = QCheckBox("Test set")
        self.test = QSpinBox()
        self.test.setRange(1, 50)
        self.test.setValue(10)
        self.test.setSuffix(" % test")
        self.test.setEnabled(False)
        self.use_test.toggled.connect(self.test.setEnabled)
        self.train_lbl = QLabel("")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        self.seed.setPrefix("seed ")
        self.rare = QSpinBox()
        self.rare.setRange(0, 50)
        self.rare.setValue(3)
        self.rare.setPrefix("rare ≤ ")
        self.rare.setSuffix(" images → train only")
        for wdg in (self.val, self.use_test, self.test, self.train_lbl, self.seed, self.rare):
            ratio.addWidget(wdg)
        ratio.addStretch(1)
        w = QWidget()
        w.setLayout(ratio)
        sf.addRow("Ratios:", w)
        grp = QHBoxLayout()
        self.g_dups = QCheckBox("Duplicates / near-duplicates")
        self.g_dups.setChecked(True)
        self.g_dups.setToolTip("Uses the image hashes cached by Duplicate Images / Stats (run it first for best results)")
        self.g_video = QCheckBox("Frames of the same video")
        self.g_video.setChecked(True)
        self.g_video.setToolTip("file_0001.jpg, file_0002.jpg … in one folder are kept in one split")
        self.g_ident = QCheckBox("Same text identity (OCR)")
        ocr = self.app.open_dialogs.get("ocr_check")
        self.g_ident.setEnabled(bool(ocr and getattr(ocr, "ctx", None) and ocr.ctx.get("reads")))
        self.g_ident.setChecked(self.g_ident.isEnabled())
        self.g_ident.setToolTip("Needs an OCR Check scan in this session (plates / container IDs kept together)")
        for c in (self.g_dups, self.g_video, self.g_ident):
            grp.addWidget(c)
        grp.addStretch(1)
        w = QWidget()
        w.setLayout(grp)
        sf.addRow("Keep together:", w)
        for wdg in (self.val, self.test, self.seed, self.rare):
            wdg.valueChanged.connect(lambda _v: self._stale())
        for c in (self.use_test, self.keep, self.new, self.g_dups, self.g_video, self.g_ident):
            c.toggled.connect(lambda _on: self._stale())
        self.keep.toggled.connect(lambda on: [x.setEnabled(not on) for x in (self.val, self.use_test, self.seed,
                                                                             self.rare, self.g_dups, self.g_video)])
        self.keep.toggled.emit(self.keep.isChecked())

        # ---------------------------------------------------------- 3 · preview
        pv_box = QGroupBox("3 · Preview")
        pl = QVBoxLayout(pv_box)
        self.bar = SplitBar()
        legend = QLabel(" · ".join(f"<span style='color:{SPLIT_COLORS[s].name()}'>■</span> {s}" for s in ORDER)
                        + " &nbsp; — dashed lines: the target ratio. A class bar close to the dashes is balanced.")
        legend.setTextFormat(Qt.TextFormat.RichText)
        apply_style(legend, "color:@muted;")
        self.class_bars = ClassBars()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.class_bars)
        self.ftable = QTableWidget(0, 7)
        self.ftable.setHorizontalHeaderLabels(["Feature", "Value", "Images", "Train", "Val", "Test", "Status"])
        self.ftable.verticalHeader().setVisible(False)
        self.ftable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ftable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.TextFormat.RichText)
        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel("<b>Per class</b> (images)"))
        tl.addWidget(scroll, 1)
        split.addWidget(top)
        split.addWidget(self.ftable)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        pl.addWidget(self.bar)
        pl.addWidget(legend)
        pl.addWidget(self.notes)
        pl.addWidget(split, 1)

        foot = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.preview_btn = newButton("Preview", icon="eye", slot=self.preview)
        self.export_btn = newButton("Export", icon="tool-export", slot=self.do_export)
        self.export_btn.setDefault(True)
        self.stop_btn = newButton("Stop", icon="delete", slot=self.cancel.set)
        self.stop_btn.setVisible(False)
        foot.addWidget(self.progress, 1)
        foot.addWidget(self.status, 2)
        for b in (self.preview_btn, self.stop_btn, self.export_btn):
            foot.addWidget(b)
        foot.addWidget(newButton("Close", slot=self.close))

        lay = QVBoxLayout(self)
        lay.addWidget(out_box)
        lay.addWidget(sp_box)
        lay.addWidget(pv_box, 1)
        lay.addLayout(foot)
        self._update_train_lbl()
        self.val.valueChanged.connect(lambda _v: self._update_train_lbl())
        self.test.valueChanged.connect(lambda _v: self._update_train_lbl())
        self.use_test.toggled.connect(lambda _on: self._update_train_lbl())
        if len(self.session.samples) <= 30000:
            self.preview()

    # ------------------------------------------------------------------ helpers
    def _update_train_lbl(self):
        t = self.test.value() if self.use_test.isChecked() else 0
        tr = 100 - self.val.value() - t
        self.train_lbl.setText(f"→ train {tr} %" + ("" if tr >= 50 else " ⚠"))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export folder", self.out.text())
        if d:
            self.out.setText(d)

    def _stale(self):
        if self.plan is not None:
            self.status.setText("Settings changed — Preview again (Export re-plans automatically).")
        self.plan = None

    def _busy(self, on, text=""):
        self.progress.setVisible(on)
        self.stop_btn.setVisible(on)
        self.preview_btn.setEnabled(not on)
        self.export_btn.setEnabled(not on)
        if on:
            self.progress.setRange(0, 0)
        self.status.setText(text)

    def _samples(self):
        s = self.session
        keys = []
        broken = set()
        if self.ex_broken.isChecked():
            for tool in ("integrity", "image_quality"):
                for k, lst in s.issues.by_tool.get(tool, {}).items():
                    if any(i.severity == "error" and not i.text.startswith(("EXIF rotation mismatch", "size mismatch"))
                           for i in lst):
                        broken.add(k)
        for x in s.samples:
            if self.ex_flag.isChecked() and s.review.status(x.key) == REVIEW_FLAG:
                continue
            if x.key in broken:
                continue
            if self.ex_empty.isChecked() and not s.index.get(x.key):
                continue
            keys.append(x)
        return keys, len(s.samples) - len(keys)

    def _class_names(self):
        s = self.session
        totals = s.label_totals()
        if s.is_yolo and not getattr(s.adapter, "mapping", None):
            base = list(s.base_adapter.class_names)
            return base + sorted(l for l in totals if l not in base)
        return sorted(totals, key=str.lower)

    # ------------------------------------------------------------------ plan
    def _build(self, samples, settings, progress):
        s = self.session
        ad = s.adapter
        cache = s.cache
        cls = s.is_classification
        items, links = [], []
        hashes, shas = [], {}
        for i, x in enumerate(samples):
            if self.cancel.is_set():
                raise RuntimeError("cancelled")
            try:
                ann = ad.load(x, light=True)
            except Exception:
                ann = None
            img = (ann.image_path if ann else None) or x.image_path
            meta = cache.get_image(img) if (cache is not None and img is not None) else None
            light = (meta or {}).get("light")
            if cls:
                feats = classification_features(ann.tags if ann else [], x.folder, light)
                labels = Counter(ann.tags) if ann else Counter()
            else:
                labels = ann.label_counts() if ann else Counter()
                boxes = []
                if ann is not None and ann.shapes:
                    W, H = (1.0, 1.0) if ann.normalized else (ann.width or 0, ann.height or 0)
                    if not (W and H):
                        sz = ad.image_size(x, ann)
                        W, H = sz if sz else (0, 0)
                    if W and H:
                        for sh in ann.shapes:
                            pts = sh.get("points") or []
                            if len(pts) >= 2:
                                xs, ys = [p[0] / W for p in pts], [p[1] / H for p in pts]
                                boxes.append((min(xs), min(ys), max(xs), max(ys)))
                feats = detection_features(labels, boxes, light)
            items.append(SplitItem(x.key, feats, labels=labels))
            if settings["dups"] and meta:
                if meta.get("sha"):
                    shas.setdefault(meta["sha"], []).append(x.key)
                if meta.get("phash"):
                    hashes.append((x.key, meta["phash"]))
            if settings["video"]:
                p = Path(img or x.key)
                sid = sequence_id(p.stem, str(p.parent))
                if sid:
                    links.append(("seq:" + sid, x.key))
            if progress and i % 500 == 0:
                progress(i, len(samples))
        for ks in shas.values():
            links += [(ks[0], k) for k in ks[1:]]
        if hashes:
            try:
                import numpy as np
                from AIO_QC.core.profile import near_duplicate_pairs
                arr = [np.frombuffer(h, dtype=np.uint8) for _k, h in hashes]
                for a, b, _d in near_duplicate_pairs(arr, 6):
                    links.append((hashes[a][0], hashes[b][0]))
            except Exception:
                pass
        if settings["ident"]:
            from AIO_QC.core.validators import identity_key
            ocr = self.app.open_dialogs.get("ocr_check")
            reads = ocr.ctx.get("reads", []) if ocr is not None and getattr(ocr, "ctx", None) else []
            for r in reads:
                t = (r.res or {}).get("text") or ""
                if float((r.res or {}).get("conf") or 0) >= 0.6 and len(t) >= 4:
                    links.append(("id:" + identity_key(t), r.key))
        keys = [it.key for it in items]
        groups = build_groups(keys + sorted({a for a, _b in links if a.startswith(("seq:", "id:"))}), links)
        for it in items:
            it.group = groups[it.key]
        return items

    def _settings(self):
        return {"keep": self.keep.isChecked(), "val": self.val.value() / 100,
                "test": (self.test.value() / 100) if self.use_test.isChecked() else 0.0, "seed": self.seed.value(),
                "rare": self.rare.value(), "dups": self.g_dups.isChecked(), "video": self.g_video.isChecked(),
                "ident": self.g_ident.isChecked() and self.g_ident.isEnabled()}

    def _plan_fn(self, then=None):
        samples, left_out = self._samples()
        st = self._settings()
        self.cancel.clear()

        def work():
            items = self._build(samples, st, None)
            if st["keep"]:
                norm = {"valid": "val", "validation": "val"}
                by = {x.key: norm.get(x.split, x.split) or "train" for x in samples}
                return items, type("P", (), {"assign": by, "report": make_report(items, by)})(), samples, left_out, st
            p = plan_split(items, st["val"], st["test"], st["seed"], st["rare"])
            return items, p, samples, left_out, st
        return work

    def preview(self, then=None):
        self._busy(True, "Analysing the dataset …")
        w = FuncWorker(self._plan_fn(), self)
        w.done.connect(lambda res: self._planned(res, then))
        w.failed.connect(lambda m: (self._busy(False, f"⚠ {m}")))
        w.start()
        self.worker = w

    def _planned(self, res, then=None):
        items, plan, samples, left_out, st = res
        self.items, self.plan, self.samples, self.left_out, self.settings = items, plan, samples, left_out, st
        rep = plan.report
        self._busy(False, "")
        self.bar.set_counts(rep["splits"])
        target = {"val": st["val"], "test": st["test"]}
        target["train"] = 1 - st["val"] - st["test"]
        if st["keep"]:
            tot = max(sum(rep["splits"].values()), 1)
            target = {k: v / tot for k, v in rep["splits"].items()}
        self.class_bars.set_data(rep["classes"], target)
        rows = sorted(rep["features"], key=lambda r: (r["status"] == "ok", r["kind"], -r["images"]))
        self.ftable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for c, v in enumerate((r["kind"], r["value"], r["images"], r["train"], r["val"], r["test"], r["status"])):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole, v)
                if c == 6:
                    it.setForeground(theme_color("ok" if v == "ok" else "muted" if "rare" in v and "IN" not in v
                                                 else "warn" if v.startswith("missing") else "error"))
                self.ftable.setItem(i, c, it)
        g = rep["groups"]
        notes = [f"{rep['total']:,} images" + (f" ({left_out:,} left out)" if left_out else "")]
        if g["multi"]:
            notes.append(f"{g['multi']:,} groups kept together (largest {g['largest']} images)")
        if g["split_across"]:
            notes.append(f"<span style='color:{hexc('error')}'>⚠ {g['split_across']:,} groups are split across "
                         "sets — duplicates / frames of one video leak between train and val</span>")
        rare = sum(1 for r in rep["features"] if r["status"].startswith("train only"))
        if rare:
            notes.append(f"{rare} rare feature(s) kept in train")
        bad = sum(1 for r in rep["features"] if r["status"] in ("MISSING IN TRAIN", "RARE IN VAL/TEST"))
        if bad:
            notes.append(f"<span style='color:{hexc('error')}'>{bad} feature(s) not covered by train</span>")
        self.notes.setText(" · ".join(notes))
        if then:
            then()

    # ------------------------------------------------------------------ export
    def do_export(self):
        if self.plan is None:
            self.preview(then=self.do_export)
            return
        out = Path(self.out.text().strip())
        if not str(out) or out.resolve() == self.session.root.resolve() or self.session.root.resolve() in out.resolve().parents:
            self.status.setText("⚠ Choose an export folder outside the dataset.")
            return
        if out.exists() and any(out.iterdir()):
            from PyQt6.QtWidgets import QMessageBox
            if QMessageBox.question(self, self.TITLE, f"{out} is not empty. Export into it anyway?\n"
                                                      "(Existing files with the same names get a numeric suffix.)") \
                    != QMessageBox.StandardButton.Yes:
                return
        assign = self.plan.assign
        by_split = {sp: [x for x in self.samples if assign.get(x.key) == sp] for sp in ORDER}
        fmt, mode = self.fmt.currentData(), self.mode.currentData()
        names = self._class_names()
        s = self.session
        self.cancel.clear()
        self._busy(True, "Exporting …")

        def work():
            stats = EX.export(by_split, s.adapter, s.root, out, fmt, names, mode, progress=None, cancel=self.cancel,
                              is_classification=s.is_classification)
            if not stats.get("cancelled"):
                EX.write_reports(out, self.plan.report, assign, {**self.settings, "format": fmt, "classes": names}, stats)
            return stats

        w = FuncWorker(work, self)
        w.done.connect(lambda st: self._exported(st, out))
        w.failed.connect(lambda m: self._busy(False, f"⚠ Export failed: {m}"))
        w.start()
        self.worker = w

    def _exported(self, stats, out):
        self._busy(False)
        if stats.get("cancelled"):
            self.status.setText("Export stopped — the folder holds a partial export.")
            return
        n = sum(stats["images"].values())
        msg = (f"Exported {n:,} images ({', '.join(f'{k} {v:,}' for k, v in stats['images'].items())})"
               + (f" · {len(stats['skipped'])} skipped" if stats["skipped"] else "")
               + (f" · {sum(stats['unknown_labels'].values())} objects with labels outside the class list"
                  if stats["unknown_labels"] else ""))
        self.status.setText(msg)
        self.app.toast.show_message(f"<b>Export finished</b><br>{msg}", kind="success",
                                    actions=[("Open folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(out))))])

    def closeEvent(self, e):
        self.cancel.set()
        if self.worker is not None:
            self.worker.wait(3000)
        self.app.forget_dialog(self)
        super().closeEvent(e)
