# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Convert & Split (QC drawer ▸ Review, Ctrl+Shift+X; Ctrl+Alt+Y opens it on YOLO) — one window that writes the open
dataset into another folder, in any format, as one set / with its own splits / with a new balanced split. The dataset
itself is never changed.

A side pane on the left walks through the steps and shows what is chosen in each (a dot marks a step that needs a
look); under it, a live summary of what will be written. One page at a time on the right:

  Output    format (YOLO detection / segmentation — suggested from the shapes —, LabelMe, COCO, masks, VOC, CVAT,
            YOLO OBB, DOTA, YOLO pose, crops, class folders …), the folder (checked live: new, empty, an earlier run to
            replace, not empty, inside the dataset), hard links or copies
  Classes   tick to include, drag (Alt+Up / Alt+Down) to order — the order is the class ids; write a class under
            another name or merge classes; load the order from a model's data.yaml / classes.txt. Every format uses
            this list (core.export.ClassView), a classifier's classes too
  Split     one set · the dataset's splits · a new split that keeps duplicates, video frames and OCR identities
            together and rare cases in train — previewed per class and per feature
  Files     sub-folders kept, or flattened with names made unique (the naming engine, renamed.csv)
  Options   flagged / broken images, background images, joined pieces; more: case variants, small objects, images
            with a class not written, broken shapes, parts that must lie inside a whole, verify

YOLO detection / segmentation run core.yolo_convert (threads, verify, re-encoding, parts); every other format
core.export (annotations read ahead and images placed by thread pools). Both write a manifest, so a second run into
the folder replaces exactly what the first wrote, and Undo removes it. Choices are remembered per dataset.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import QObject, QRect, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QSpinBox, QSplitter, QStackedWidget, QStyle, QStyledItemDelegate, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from AIO_QC import utils
from AIO_QC.core import export as EX
from AIO_QC.core import yolo_convert as YC
from AIO_QC.core.dataset import DatasetType
from AIO_QC.core.state import REVIEW_FLAG
from .inference_service import FuncWorker
from AIO_QC.core.diag import ignored as _ignored

ROLE_NAME = Qt.ItemDataRole.UserRole
ROLE_OBJ = Qt.ItemDataRole.UserRole + 1
ROLE_IMG = Qt.ItemDataRole.UserRole + 2
ROLE_TYPES = Qt.ItemDataRole.UserRole + 3
ROLE_ID = Qt.ItemDataRole.UserRole + 4
ROLE_OUT = Qt.ItemDataRole.UserRole + 5        # the class it is written as ("" / same name: its own)
ROLE_NAV_KEY = Qt.ItemDataRole.UserRole + 10
ROLE_NAV_SUB = Qt.ItemDataRole.UserRole + 11
ROLE_NAV_STATE = Qt.ItemDataRole.UserRole + 12
TYPE_WORD = {"rectangle": "boxes", "polygon": "polygons", "circle": "circles", "oriented rectangle": "oriented boxes",
             "mask": "masks", "line": "lines", "linestrip": "lines", "point": "points", "points": "points"}
ONE = {"objects": "object", "labels": "label", "images": "image", "classes": "class", "background images": "background image",
       "lines / points": "line / point", "boxes": "box", "polygons": "polygon", "circles": "circle", "oriented boxes": "oriented box", "masks": "mask",
       "lines": "line", "points": "point"}


def _n(n: int, word: str) -> str:
    """'1 box', '3 boxes'."""
    return f"{n:,} {ONE.get(word, word) if n == 1 else word}"


def scrolled_fixed(w, max_h: int):
    from PyQt6.QtWidgets import QScrollArea
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(w)
    sa.setMaximumHeight(max_h)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    w.setMinimumWidth(0)
    sa.setFrameShape(QScrollArea.Shape.NoFrame)
    return sa


class _Bus(QObject):
    step = pyqtSignal(int, int, str)


class _ClassRow(QStyledItemDelegate):
    """[x]  3  car ……… 1,234 objects · 560 images      (unticked rows are dimmed and have no id)"""

    def sizeHint(self, opt, index):
        return QSize(opt.rect.width(), 40)

    def paint(self, p, opt, index):
        self.initStyleOption(opt, index)
        on = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
        cid = index.data(ROLE_ID)
        name = index.data(ROLE_NAME)
        objs, imgs = index.data(ROLE_OBJ) or 0, index.data(ROLE_IMG) or 0
        types = index.data(ROLE_TYPES) or {}
        opt.text = ""
        w = opt.widget
        style = w.style() if w is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, p, w)
        r = opt.rect.adjusted(34, 0, -10, 0)
        p.save()
        ink = QColor(utils.hexc("text"))
        muted = QColor(utils.hexc("muted"))
        if not on:
            ink = muted
        f = p.font()
        top = QRect(r.left(), r.top() + 3, r.width(), r.height() // 2)
        bot = QRect(r.left(), r.top() + r.height() // 2 - 1, r.width(), r.height() // 2)
        idw = 34
        if on and cid is not None:
            fm = f
            fm.setBold(True)
            p.setFont(fm)
            p.setPen(QColor(utils.hexc("accent")))
            p.drawText(QRect(r.left(), top.top(), idw - 6, top.height()), int(Qt.AlignmentFlag.AlignVCenter
                                                                               | Qt.AlignmentFlag.AlignRight), str(cid))
            f.setBold(False)
        p.setFont(f)
        p.setPen(ink)
        out = index.data(ROLE_OUT) or name
        shown = name if out == name else f"{name}  →  {out}"
        p.drawText(QRect(r.left() + idw, top.top(), r.width() - idw, top.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   p.fontMetrics().elidedText(shown, Qt.TextElideMode.ElideRight, r.width() - idw - 150))
        p.setPen(muted)
        right = f"{_n(objs, 'objects')} · {_n(imgs, 'images')}" if objs else "not in this dataset (kept for its id)"
        p.drawText(top, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), right)
        if types:
            words = Counter()
            for t, n in types.items():
                words[TYPE_WORD.get(t, t)] += n
            txt = " · ".join(_n(n, t) for t, n in words.most_common())
            sf = p.font()
            sf.setPointSizeF(max(7.0, sf.pointSizeF() - 1))
            p.setFont(sf)
            p.drawText(QRect(bot.left() + idw, bot.top(), bot.width() - idw, bot.height()),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), txt)
        p.restore()


class _NavRow(QStyledItemDelegate):
    """Side pane row: a dot (ok / warn / error), the step in bold, what is chosen in it below (muted)."""

    def sizeHint(self, opt, index):
        return QSize(opt.rect.width(), 50)

    def paint(self, p, opt, index):
        self.initStyleOption(opt, index)
        title = opt.text
        opt.text = ""
        opt.icon = QIcon()
        w = opt.widget
        style = w.style() if w is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, p, w)
        enabled = bool(index.flags() & Qt.ItemFlag.ItemIsEnabled)
        r = opt.rect.adjusted(12, 6, -10, -6)
        p.save()
        p.setRenderHint(p.RenderHint.Antialiasing)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon):
            icon.paint(p, QRect(r.left(), r.top() + 2, 18, 18))
        state = index.data(ROLE_NAV_STATE) or ""
        if state:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(utils.hexc({"ok": "ok", "warn": "warn", "error": "error"}.get(state, "muted"))))
            p.drawEllipse(r.right() - 8, r.top() + 7, 8, 8)
        f = p.font()
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(utils.hexc("text" if enabled else "muted")))
        tx = r.left() + 28
        p.drawText(QRect(tx, r.top(), r.right() - tx - 12, 20), int(Qt.AlignmentFlag.AlignVCenter), title)
        f.setBold(False)
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        p.setPen(QColor(utils.hexc("muted")))
        sub = index.data(ROLE_NAV_SUB) or ""
        p.drawText(QRect(tx, r.top() + 20, r.right() - tx, 18), int(Qt.AlignmentFlag.AlignVCenter),
                   p.fontMetrics().elidedText(sub, Qt.TextElideMode.ElideRight, r.right() - tx))
        p.restore()


SECTIONS = (("output", "Output", "tool-convert"), ("classes", "Classes", "list"), ("split", "Split", "tool-splits"),
            ("files", "Files", "folder-line"), ("options", "Options", "settings"))
FMT_SUFFIX = {"yolo_det": "_yolo", "yolo_seg": "_yolo", "labelme": "_labelme", "coco": "_coco", "masks": "_masks",
              "voc": "_voc", "cvat": "_cvat", "yolo_obb": "_obb", "dota": "_dota", "yolo_pose": "_pose",
              "crops": "_crops", "flags_folders": "_classes", "cls_folders": "_classes", "cls_csv": "_csv"}
YOLO_FORMATS = ("yolo_det", "yolo_seg")
SHORT = {"yolo_det": "YOLO detection", "yolo_seg": "YOLO segmentation", "labelme": "LabelMe", "coco": "COCO",
         "masks": "Masks (PNG)", "voc": "Pascal VOC", "cvat": "CVAT", "yolo_obb": "YOLO OBB", "dota": "DOTA",
         "yolo_pose": "YOLO pose", "crops": "Crops", "flags_folders": "Flags → class folders",
         "cls_folders": "Class folders", "cls_csv": "Class table (CSV)"}
OPENABLE = {"yolo_det", "yolo_seg", "labelme", "coco", "cls_folders", "crops", "flags_folders"}


class ConvertSplitDialog(QDialog):
    TITLE = "Convert & Split"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        s = self.session
        self.cls = bool(s.is_classification)
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(utils.newIcon("tool-convert"))
        from AIO_QC.utils.qt_utils import fit_size, scrolled
        fit_size(self, "large")
        self.survey: YC.Survey | None = None
        self.cancel = threading.Event()
        self.worker = None
        self.bus = _Bus(self)
        self.bus.step.connect(self._on_step)
        self._t0 = 0.0
        self._last_out = None
        self._last_fmt = ""
        self._samples_memo = None
        self._key = "convert_yolo/" + str(s.root)
        try:
            self.saved = json.loads(utils.app_settings().value(self._key, "") or "{}")
        except ValueError:
            self.saved = {}

        intro = QLabel("Writes this dataset into another folder — as YOLO or another format, as one set, with its "
                       "own splits or a new balanced train / val / test split. The dataset itself is not changed.")
        intro.setWordWrap(True)
        utils.style(intro, "QLabel{color:@muted;}")

        # ------------------------------------------------------------------ output: format
        fmt_box = QGroupBox("Format")
        fl = QVBoxLayout(fmt_box)
        self.fmt = QComboBox()
        for f in self._formats():
            self.fmt.addItem(EX.FORMATS[f], f)
        want = self.saved.get("format") or {"seg": "yolo_seg", "det": "yolo_det"}.get(self.saved.get("task", ""), "")
        if not want:
            want = "cls_folders" if self.cls else "yolo_det"
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(want)))
        self.task_hint = QLabel("" if self.cls else "Looking at the shapes …")
        self.task_hint.setWordWrap(True)
        self.task_hint.setTextFormat(Qt.TextFormat.RichText)
        utils.style(self.task_hint, "QLabel{color:@muted;}")
        crow = QHBoxLayout()
        crow.setContentsMargins(0, 0, 0, 0)
        crop = self.saved.get("crop") or {}
        self.crop_pad = QDoubleSpinBox()
        self.crop_pad.setRange(0, 200)
        self.crop_pad.setValue(float(crop.get("pad", 10)))
        self.crop_pad.setSuffix(" % margin")
        self.crop_square = QCheckBox("Square")
        self.crop_square.setChecked(bool(crop.get("square", False)))
        self.crop_min = QSpinBox()
        self.crop_min.setRange(1, 4096)
        self.crop_min.setValue(int(crop.get("min", 8)))
        self.crop_min.setPrefix("skip below ")
        self.crop_min.setSuffix(" px")
        for x in (self.crop_pad, self.crop_square, self.crop_min):
            crow.addWidget(x)
        crow.addStretch(1)
        self.crop_row = QWidget()
        self.crop_row.setLayout(crow)
        self.crop_row.setToolTip("Every shape's box (plus the margin) becomes one image in <split>/<class>/ — a "
                                 "classification dataset made from your boxes")
        self.fmt_note = QLabel("")
        self.fmt_note.setWordWrap(True)
        utils.style(self.fmt_note, "QLabel{color:@muted;}")
        fl.addWidget(self.fmt)
        fl.addWidget(self.task_hint)
        fl.addWidget(self.crop_row)
        fl.addWidget(self.fmt_note)

        # ------------------------------------------------------------------ output: save to
        out_box = QGroupBox("Save to")
        ol = QVBoxLayout(out_box)
        r = QHBoxLayout()
        self.out = QLineEdit(self.saved.get("out") or self._default_out(want))
        self.out.textChanged.connect(lambda _t: (self._check_out(), self._refresh_summary()))
        r.addWidget(self.out, 1)
        r.addWidget(utils.newButton("Browse…", icon="open", slot=self._browse))
        self.out_state = QLabel("")
        self.out_state.setWordWrap(True)
        self.out_state.setTextFormat(Qt.TextFormat.RichText)
        self.out_fix = QPushButton("")
        self.out_fix.setVisible(False)
        self.out_fix.clicked.connect(self._use_subfolder)
        sr = QHBoxLayout()
        sr.addWidget(self.out_state, 1)
        sr.addWidget(self.out_fix)
        ol.addLayout(r)
        ol.addLayout(sr)
        self.link = QComboBox()
        self.link.addItem("Hard links (no extra disk space; copies when not possible)", "hardlink")
        self.link.addItem("Copies", "copy")
        self.link.setCurrentIndex(max(0, self.link.findData(self.saved.get("link", "hardlink"))))
        self.link.setToolTip("A hard link is the same file under a second name: no extra space, written in an instant "
                             "— on the same drive. Another drive (or a file system without links) gets copies.")
        lr = QHBoxLayout()
        lr.addWidget(QLabel("Images:"))
        lr.addWidget(self.link, 1)
        ol.addLayout(lr)
        mapping = getattr(s.adapter, "mapping", None)
        if mapping:
            ml = QLabel(f"The label mapping is applied ({len(mapping)} label(s) renamed / merged).")
            utils.style(ml, "QLabel{color:@muted;}")
            ol.addWidget(ml)

        # ------------------------------------------------------------------ classes
        cls_box = QGroupBox("Classes")
        cl = QVBoxLayout(cls_box)
        self.cls_hint = QLabel("Tick the classes to write and drag them into order — the order is the class id "
                               "(0, 1, 2 …). To match a model trained before, load its class list. Right-click a "
                               "class to write it under another name or merge it into another class.")
        self.cls_hint.setWordWrap(True)
        utils.style(self.cls_hint, "QLabel{color:@muted;}")
        self.list = QListWidget()
        self.list.setItemDelegate(_ClassRow(self.list))
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setUniformItemSizes(True)
        self.list.model().rowsMoved.connect(lambda *_: self._renumber())
        self.list.itemChanged.connect(lambda *_: self._renumber())
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._class_menu)
        self.list.setToolTip("Right-click (or F2): write a class under another name, or merge it into another class")
        QShortcut(QKeySequence("F2"), self.list, self._rename_selected)
        QShortcut(QKeySequence("Alt+Up"), self.list, lambda: self._move(-1))
        QShortcut(QKeySequence("Alt+Down"), self.list, lambda: self._move(1))
        row1 = QHBoxLayout()
        self.sort_box = QComboBox()
        self.sort_box.addItem("Sort…", "")
        self.sort_box.addItem("A – Z", "az")
        self.sort_box.addItem("Most objects first", "count")
        self.sort_box.addItem("Ticked first", "ticked")
        self.sort_box.activated.connect(self._sort)
        load = utils.newButton("Load order from file…", icon="open", slot=self._load_order,
                               tip="data.yaml / dataset.yaml / classes.txt / *.names of a model or another dataset")
        up = utils.newButton("", icon="chevron-up", slot=lambda: self._move(-1), tip="Move up (Alt+Up)")
        down = utils.newButton("", icon="chevron-down", slot=lambda: self._move(1), tip="Move down (Alt+Down)")
        for b in (up, down):
            b.setFixedWidth(34)
        row1.addWidget(self.sort_box)
        row1.addWidget(load)
        row1.addStretch(1)
        row1.addWidget(up)
        row1.addWidget(down)
        row2 = QHBoxLayout()
        self.count_lbl = QLabel("")
        row2.addWidget(self.count_lbl, 1)
        row2.addWidget(utils.newButton("All", slot=lambda: self._tick_all(True)))
        row2.addWidget(utils.newButton("None", slot=lambda: self._tick_all(False)))
        self.cls_off = QLabel("")
        self.cls_off.setWordWrap(True)
        self.cls_off.setVisible(False)
        utils.style(self.cls_off, "QLabel{color:@warn;}")
        cl.addWidget(self.cls_hint)
        cl.addWidget(self.cls_off)
        cl.addLayout(row1)
        cl.addWidget(self.list, 1)
        cl.addLayout(row2)

        # ------------------------------------------------------------------ split
        lay_box = QGroupBox("Split")
        ll = QVBoxLayout(lay_box)
        has_splits = any(x.split for x in s.samples)
        self.flat = QRadioButton("One set — no split folders")
        self.keep = QRadioButton("Keep the dataset's splits — train/ val/ test/" + ("" if has_splits else
                                                                                      " (it has none)"))
        self.keep.setEnabled(has_splits)
        self.new = QRadioButton("New balanced split")
        g2 = QButtonGroup(self)
        for b in (self.flat, self.keep, self.new):
            g2.addButton(b)
        wl = self.saved.get("layout", "splits" if has_splits else "flat")
        {"flat": self.flat, "splits": self.keep if has_splits else self.flat, "new": self.new}.get(
            wl, self.flat).setChecked(True)
        self.val = QSpinBox()
        self.val.setRange(1, 50)
        self.val.setValue(int(self.saved.get("val", 10)))
        self.val.setSuffix(" % val")
        self.test = QSpinBox()
        self.test.setRange(0, 50)
        self.test.setValue(int(self.saved.get("test", 0)))
        self.test.setSuffix(" % test")
        self.test.setToolTip("0 %: no test set")
        self.train_lbl = QLabel("")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(int(self.saved.get("seed", 42)))
        self.seed.setPrefix("seed ")
        self.seed.setToolTip("The same seed and settings give the same split every time")
        self.rare = QSpinBox()
        self.rare.setRange(0, 50)
        self.rare.setValue(int(self.saved.get("rare", 3)))
        self.rare.setPrefix("rare ≤ ")
        self.rare.setSuffix(" → train")
        self.rare.setToolTip("A class, class combination, object size, position or lighting seen in this many images or "
                             "fewer is kept in train: too few to learn from if some went to val / test")
        self.preview_btn = utils.newButton("Preview split", icon="tool-splits", slot=self.preview_split,
                                           tip="Plan the split now and see it per class (Convert uses this plan)")
        nr = QHBoxLayout()
        nr.setContentsMargins(24, 0, 0, 0)
        for wdg in (self.val, self.test, self.train_lbl):
            nr.addWidget(wdg)
        nr.addStretch(1)
        nr.addWidget(self.preview_btn)
        nr2 = QHBoxLayout()
        nr2.setContentsMargins(24, 0, 0, 0)
        for wdg in (self.seed, self.rare):
            nr2.addWidget(wdg)
        nr2.addStretch(1)
        self.g_dups = QCheckBox("Keep duplicates and look-alike images together")
        self.g_dups.setChecked(bool(self.saved.get("dups", True)))
        self.g_dups.setToolTip("Exact copies and near-identical images (perceptual hash) always go to the same split "
                               "— otherwise val scores look better than they are")
        self.g_video = QCheckBox("Keep frames of one video together")
        self.g_video.setChecked(bool(self.saved.get("video", True)))
        self.g_video.setToolTip("Files named like clip_frame_0012 (same folder and prefix, a frame number) are one "
                                "video: its frames go to one split")
        self.g_ident = QCheckBox("Keep one text identity together (plates, container IDs — from OCR Check)")
        ocr = self.app.open_dialogs.get("ocr_check")
        self.g_ident.setEnabled(bool(ocr and getattr(ocr, "ctx", None) and ocr.ctx.get("reads")))
        self.g_ident.setChecked(self.g_ident.isEnabled())
        self.g_ident.setToolTip("Needs an OCR Check scan in this session: images of the same plate / ID go to one split")
        self.g_look = QCheckBox("Look at the images: balance lighting, find look-alikes (once; then remembered)")
        self.g_look.setChecked(bool(self.saved.get("measure", True)))
        self.g_look.setToolTip("Reads each image once at reduced size (about 1 ms each) for its lighting and a "
                               "perceptual hash; results are kept in the dataset cache")
        self.split_note = QLabel("")
        self.split_note.setWordWrap(True)
        self.split_note.setTextFormat(Qt.TextFormat.RichText)
        utils.style(self.split_note, "QLabel{color:@muted;}")
        from .export_dialog import ClassBars, SplitBar
        self.split_bar = SplitBar()
        self.split_bar.setFixedHeight(64)
        self.class_bars = ClassBars()
        self.class_bars.sizeHint = lambda: QSize(300, self.class_bars.ROW * max(1, len(self.class_bars.classes)) + 8)
        self.ftable = QTableWidget(0, 7)
        self.ftable.setHorizontalHeaderLabels(["Feature", "Value", "Images", "Train", "Val", "Test", "Status"])
        self.ftable.verticalHeader().setVisible(False)
        self.ftable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ftable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.ftable.setMinimumHeight(140)
        self.ftable.setToolTip("Every feature the split balances — class, combination, size, position, lighting — "
                               "and how its images went (problems first)")
        self.preview_panel = QWidget()
        pv = QVBoxLayout(self.preview_panel)
        pv.setContentsMargins(24, 4, 0, 0)
        pv.addWidget(self.split_bar)
        pv.addWidget(QLabel("<b>Per class</b> (images) — dashed lines: the target"))
        pv.addWidget(scrolled_fixed(self.class_bars, 220))
        pv.addWidget(QLabel("<b>Per feature</b>"))
        pv.addWidget(self.ftable)
        self.preview_panel.setVisible(False)
        self.new_row = QWidget()
        nv = QVBoxLayout(self.new_row)
        nv.setContentsMargins(0, 0, 0, 0)
        nv.addLayout(nr)
        nv.addLayout(nr2)
        for wdg in (self.g_dups, self.g_video, self.g_ident, self.g_look):
            h = QHBoxLayout()
            h.setContentsMargins(24, 0, 0, 0)
            h.addWidget(wdg)
            nv.addLayout(h)
            wdg.toggled.connect(lambda _on: self._split_changed())
        h = QHBoxLayout()
        h.setContentsMargins(24, 0, 0, 0)
        h.addWidget(self.split_note)
        nv.addLayout(h)
        nv.addWidget(self.preview_panel)
        for sb in (self.val, self.test, self.seed, self.rare):
            sb.valueChanged.connect(lambda _v: self._split_changed())
        self.plan = None                                 # (signature, items, plan) of the last preview
        ll.addWidget(self.flat)
        ll.addWidget(self.keep)
        ll.addWidget(self.new)
        ll.addWidget(self.new_row)
        self.new.toggled.connect(self.new_row.setEnabled)
        self.new_row.setEnabled(self.new.isChecked())

        # ------------------------------------------------------------------ files
        files_box = QGroupBox("Files")
        fv = QVBoxLayout(files_box)
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Sub-folders:"))
        self.folders = QComboBox()
        self.folders.addItem("Keep them, as in the dataset (images/camA/…)", "keep")
        self.folders.addItem("Flatten — every image straight in images/, names made unique", "flat")
        self.folders.setToolTip("Most training tools are happy with sub-folders. Flatten when a tool wants one folder: "
                                "files with the same name in different sub-folders then get new names (checked "
                                "below, listed in renamed.csv)")
        self.folders.setCurrentIndex(max(0, self.folders.findData(self.saved.get("folders", "keep"))))
        fr.addWidget(self.folders, 1)
        self.folders_note = QLabel("")
        self.folders_note.setWordWrap(True)
        utils.style(self.folders_note, "QLabel{color:@muted;}")
        fv.addLayout(fr)
        fv.addWidget(self.folders_note)
        from .naming_panel import NamingPanel
        self.naming = NamingPanel(self, context="convert_yolo")
        self.naming.setMinimumHeight(220)
        self.naming.setVisible(False)
        self.naming.planChanged.connect(lambda _p: self._refresh_summary())
        fv.addWidget(self.naming)
        self.fix_img = QCheckBox("Re-encode images YOLO trainers can't read (GIF, AVIF …) as PNG")
        self.fix_img.setChecked(bool(self.saved.get("fix_images", True)))
        self.fix_img.setToolTip("Also: a JPEG without its end marker is copied, never linked — trainers rewrite such "
                                "files in place, which would change your original through the link")
        fv.addWidget(self.fix_img)
        self.folders.currentIndexChanged.connect(lambda _i: self._folders_changed())
        for b in (self.flat, self.keep, self.new):
            b.toggled.connect(lambda on: on and (self._folders_changed(), self._split_changed()))
        QTimer.singleShot(0, self._folders_changed)

        # ------------------------------------------------------------------ options
        opt_box = QGroupBox("What is written")
        pl = QVBoxLayout(opt_box)
        self.merge = QCheckBox("Join the pieces of one object (shapes with the same group ID and label)")
        self.merge.setChecked(bool(self.saved.get("merge", True)))
        self.merge.setToolTip("An object split by something in front of it is often drawn as several shapes with one "
                              "group ID. YOLO wants one box / polygon per object.")
        self.bg = QCheckBox("Keep images without a written class (in _unlabelled/)" if self.cls else
                            "Keep images without objects as background (empty label files)")
        self.bg.setChecked(bool(self.saved.get("background", not self.cls)))
        self.bg.setToolTip("Background images teach a detector what is NOT an object. Images whose objects are all "
                           "left out count as background too.")
        self.ex_flag = QCheckBox("Leave out flagged images")
        self.ex_flag.setChecked(bool(self.saved.get("ex_flag", True)))
        self.ex_broken = QCheckBox("Leave out broken images")
        self.ex_broken.setChecked(bool(self.saved.get("ex_broken", True)))
        self.ex_broken.setToolTip("Corrupt / truncated / undecodable images found by the background check or Image "
                                  "Quality")
        for wdg in (self.merge, self.bg, self.ex_flag, self.ex_broken):
            pl.addWidget(wdg)
            wdg.toggled.connect(lambda _on: self._refresh_summary())
        for wdg in (self.ex_flag, self.ex_broken):
            wdg.toggled.connect(lambda _on: (self._forget_samples(), self._split_changed()))

        # ------------------------------------------------------------------ more options
        adv = QGroupBox("")
        al = QVBoxLayout(adv)
        self.fold = QCheckBox("Merge labels that differ only in case or spaces (“Car”, “car ”)")
        self.fold.setChecked(bool(self.saved.get("fold", False)))
        self.min_row = QWidget()
        r = QHBoxLayout(self.min_row)
        r.setContentsMargins(0, 0, 0, 0)
        r.addWidget(QLabel("Leave out objects smaller than"))
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 100000)
        self.min_area.setValue(int(self.saved.get("min_area", 2)))
        self.min_area.setSuffix(" px²")
        self.min_area.setToolTip("Specks from a slipped click; 0 keeps everything with an area")
        r.addWidget(self.min_area)
        r.addStretch(1)
        self.left_out = QComboBox()
        self.left_out.addItem("keep the image, without those shapes", "drop")
        self.left_out.addItem("leave the image out", "skip")
        self.left_out.setToolTip("An object of a class you don't write becomes background in that image. Leave the "
                                 "image out when that class looks like one you train (it would teach the model to miss it)")
        self.broken_box = QComboBox()
        self.broken_box.addItem("keep the image, without that shape", "drop")
        self.broken_box.addItem("leave the image out", "skip")
        self.broken_box.setToolTip("Shapes that can't be read (bad coordinates, too few points). Leaving the image out "
                                   "avoids an object the model is then told isn't there.")
        for box, key in ((self.left_out, "left_out"), (self.broken_box, "broken")):
            box.setCurrentIndex(max(0, box.findData(self.saved.get(key, "drop"))))
        f1 = QHBoxLayout()
        f1.addWidget(QLabel("Images with a class that is not written:"))
        f1.addWidget(self.left_out, 1)
        self.broken_row = QWidget()
        f2 = QHBoxLayout(self.broken_row)
        f2.setContentsMargins(0, 0, 0, 0)
        f2.addWidget(QLabel("Images with a broken shape:"))
        f2.addWidget(self.broken_box, 1)
        self.verify_box = QCheckBox("Read every label back after writing (verify)")
        self.verify_box.setChecked(bool(self.saved.get("verify", True)))
        al.addWidget(self.fold)
        al.addWidget(self.min_row)
        al.addLayout(f1)
        al.addWidget(self.broken_row)
        al.addWidget(self.verify_box)
        # parts that must lie inside a whole
        self.parts_panel = QWidget()
        ppl = QVBoxLayout(self.parts_panel)
        ppl.setContentsMargins(0, 0, 0, 0)
        pt = QLabel("<b>Parts inside a whole</b> — a part outside every whole of its rule is not written "
                    "(e.g. a text line that must be on a plate). Parts don't drive the split balance.")
        pt.setWordWrap(True)
        pt.setTextFormat(Qt.TextFormat.RichText)
        ppl.addWidget(pt)
        self.parts_box = QVBoxLayout()
        ppl.addLayout(self.parts_box)
        pr = QHBoxLayout()
        pr.addWidget(utils.newButton("Add a rule", icon="new", slot=lambda: self._add_part_rule()))
        self.orphans = QComboBox()
        self.orphans.addItem("drop the part", "drop")
        self.orphans.addItem("leave the image out", "skip")
        self.orphans.setCurrentIndex(max(0, self.orphans.findData(self.saved.get("orphans", "drop"))))
        pr.addStretch(1)
        pr.addWidget(QLabel("A part outside its whole:"))
        pr.addWidget(self.orphans)
        ppl.addLayout(pr)
        al.addWidget(self.parts_panel)
        self._part_rows = []
        for wdg in (self.fold, self.fix_img, self.verify_box):
            wdg.toggled.connect(lambda _on: self._refresh_summary())
        for wdg in (self.left_out, self.broken_box, self.orphans):
            wdg.currentIndexChanged.connect(lambda _i: self._refresh_summary())
        self.min_area.valueChanged.connect(lambda _v: self._refresh_summary())
        self.fold.toggled.connect(lambda _on: self._split_changed())
        from .drawer import OptionsDrawer
        self.more = OptionsDrawer(adv, "More options", "Less options",
                                  tip="Merging case variants, small objects, image rules, parts inside a whole, …")
        if any(self.saved.get(k) for k in ("fold", "parts")) or self.saved.get("left_out", "drop") != "drop":
            self.more.set_open(True, animate=False)

        # ------------------------------------------------------------------ summary + footer
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setObjectName("convSummary")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        utils.style(self.summary, "#convSummary{background:@surface; border:1px solid @line; border-radius:8px; "
                                  "padding:10px;}")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.go = utils.newButton("Convert", icon="tool-convert", slot=self.start,
                                  tip="Write the dataset with these settings (Ctrl+Enter)")
        self.go.setProperty("variant", "primary")
        self.go.setDefault(True)
        QShortcut(QKeySequence("Ctrl+Return"), self, self.start)
        self.stop_btn = utils.newButton("Stop", icon="stop", slot=self.cancel.set,
                                        tip="Stop after the image in progress — what is done so far is kept")
        self.stop_btn.setVisible(False)
        self.undo_btn = utils.newButton("Undo", icon="undo", slot=self._undo,
                                        tip="Remove what this run wrote (only that — nothing else in the folder)")
        self.undo_btn.setVisible(False)
        self.open_btn = utils.newButton("Open folder", icon="open", slot=self._open_folder)
        self.reopen_btn = utils.newButton("Open in AIO QC", icon="tool-validation", slot=self._open_in_app,
                                          tip="Open the written dataset here to check it (this dataset closes)")
        self.train_btn = utils.newButton("Train a model…", icon="tool-train", slot=self._train,
                                         tip="Train a model on the written dataset (Models ▸ Train Model)")
        for b in (self.open_btn, self.reopen_btn, self.train_btn):
            b.setVisible(False)

        # ------------------------------------------------------------------ the window: side pane + one page
        def page(*widgets, stretch=True):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(4, 0, 4, 0)
            for x in widgets:
                v.addWidget(x)
            if stretch:
                v.addStretch(1)
            return scrolled(w)

        self.pages = QStackedWidget()
        cls_page = QWidget()
        cv = QVBoxLayout(cls_page)
        cv.setContentsMargins(4, 0, 4, 0)
        cv.addWidget(cls_box)
        for wdg in (page(fmt_box, out_box), cls_page, page(lay_box), page(files_box), page(opt_box, self.more)):
            self.pages.addWidget(wdg)
        self.nav = QListWidget()
        self.nav.setObjectName("convNav")
        self.nav.setItemDelegate(_NavRow(self.nav))
        self.nav.setUniformItemSizes(True)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        utils.style(self.nav, "#convNav{background:transparent; border:none;} "
                              "#convNav::item{border-radius:8px; margin:1px 0;} "
                              "#convNav::item:selected{background:@surface; border:1px solid @line;}")
        for i, (key, title, icon) in enumerate(SECTIONS):
            it = QListWidgetItem(utils.newIcon(icon), title)
            it.setData(ROLE_NAV_KEY, key)
            it.setToolTip(f"{title} (Alt+{i + 1})")
            self.nav.addItem(it)
            QShortcut(QKeySequence(f"Alt+{i + 1}"), self, lambda i=i: self.nav.setCurrentRow(i))
        self.nav.setFixedHeight(52 * len(SECTIONS) + 6)
        self.nav.currentRowChanged.connect(self._section_changed)
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.addWidget(self.nav)
        what = QLabel("<b>What will be written</b>")
        sv.addWidget(what)
        sum_scroll = scrolled(self.summary)
        sv.addWidget(sum_scroll, 1)
        side.setMinimumWidth(260)
        side.setMaximumWidth(380)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(side)
        split.addWidget(self.pages)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 900])
        split.setChildrenCollapsible(False)
        foot = QHBoxLayout()
        foot.addWidget(self.progress, 2)
        foot.addWidget(self.status, 3)
        for b in (self.undo_btn, self.open_btn, self.reopen_btn, self.train_btn, self.stop_btn, self.go):
            foot.addWidget(b)
        foot.addWidget(utils.newButton("Close", slot=self.close))
        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addWidget(split, 1)
        lay.addLayout(foot)

        self._summary_timer = QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.setInterval(250)
        self._summary_timer.timeout.connect(self._summary_now)
        self._fill_classes()
        for part, whole, pct in self.saved.get("parts", []) or []:
            self._add_part_rule(part, whole, pct)
        self.fmt.currentIndexChanged.connect(lambda _i: self._fmt_changed())
        self._fmt_changed(first=True)
        self._check_out()
        self._split_changed()
        self.nav.setCurrentRow(0)
        self._start_survey()

    # ================================================================== format / sections
    def _formats(self) -> list[str]:
        if self.cls:
            return ["cls_folders", "labelme", "cls_csv"]
        return ["yolo_det", "yolo_seg", "labelme", "coco", "masks", "voc", "cvat", "yolo_obb", "dota", "yolo_pose",
                "crops"] + (["flags_folders"] if self.session.type == DatasetType.LABELME else [])

    def fmt_key(self) -> str:
        return self.fmt.currentData() or ""

    def is_yolo(self) -> bool:
        """YOLO detection / segmentation: the full converter (parts, joined pieces, verify, replace / undo …)."""
        return self.fmt_key() in YOLO_FORMATS and not self.cls

    def _yolo_task(self) -> str:
        return "seg" if self.fmt_key() == "yolo_seg" else "det"

    def uses_classes(self) -> bool:
        return self.fmt_key() != "flags_folders"

    def _source_fmt(self) -> str:
        t = self.session.type
        if t in (DatasetType.YOLO, DatasetType.YOLO_SPLIT):
            return "yolo_seg" if self.survey is not None and self.survey.suggested_task() == "seg" else "yolo_det"
        return {DatasetType.LABELME: "labelme", DatasetType.COCO: "coco", DatasetType.CLS_FOLDERS: "cls_folders",
                DatasetType.CLS_LABELME: "labelme"}.get(t, "")

    def _default_out(self, fmt: str) -> str:
        root = self.session.root
        same = fmt == self._source_fmt() or (fmt in YOLO_FORMATS and self.session.is_yolo)
        return str(root.parent / f"{root.name}{'_split' if same else FMT_SUFFIX.get(fmt, '_out')}")

    def _is_default_out(self, txt: str) -> bool:
        root = self.session.root
        names = {f"{root.name}{x}" for x in set(FMT_SUFFIX.values()) | {"_split", "_out"}}
        p = Path(txt.strip() or ".")
        return p.parent == root.parent and p.name in names

    def _fmt_changed(self, first=False):
        f, y = self.fmt_key(), self.is_yolo()
        self.crop_row.setVisible(f == "crops")
        notes = {"labelme": "One JSON per image beside it — every shape type kept as drawn.",
                 "coco": "annotations/instances_<split>.json; boxes and polygons (segmentation).",
                 "masks": "A class-index PNG and an instance PNG per image (semantic / instance segmentation).",
                 "voc": "Pascal VOC XML per image (boxes) + ImageSets.",
                 "cvat": "One CVAT for images 1.1 annotations.xml (every split a subset).",
                 "yolo_obb": "Oriented boxes: an oriented rectangle as drawn, any other shape's tightest rotated box.",
                 "dota": "DOTA labelTxt (oriented boxes in pixels).",
                 "yolo_pose": "Boxes as classes, point shapes as their keypoints (kpt_shape and flip_idx written).",
                 "crops": "Every object cut out into <split>/<class>/ — a classification dataset from your boxes.",
                 "flags_folders": "The image flags become the classes — every image in its flag's folder.",
                 "cls_folders": "<split>/<class>/<image> — what classifier trainers read.",
                 "cls_csv": "<split>/images/ and labels.csv (file, classes, split)."}
        self.fmt_note.setText(notes.get(f, ""))
        self.fmt_note.setVisible(bool(notes.get(f)))
        self.task_hint.setVisible(not self.cls)
        for w in (self.merge, self.fix_img, self.min_row, self.broken_row, self.verify_box, self.parts_panel):
            w.setVisible(y)
        flat_ok = f not in EX.CLS_FORMATS and f != "crops"
        self.folders.setEnabled(flat_ok)
        if not flat_ok and self.folders.currentData() == "flat":
            self.folders.setCurrentIndex(0)
        self.folders_note.setText("" if flat_ok else "Class folders keep their own layout (<split>/<class>/).")
        self.folders_note.setVisible(not flat_ok)
        uses = self.uses_classes()
        self.cls_off.setVisible(not uses)
        self.cls_off.setText("This format writes the image flags as the classes — the class list is not used.")
        self.list.setEnabled(uses)
        it = self.nav.item(1) if self.nav.count() > 1 else None
        if it is not None:
            it.setFlags((it.flags() | Qt.ItemFlag.ItemIsEnabled) if uses else (it.flags() & ~Qt.ItemFlag.ItemIsEnabled))
        self.go.setText("Convert")
        if not first and self._is_default_out(self.out.text()):
            self.out.setText(self._default_out(f))
        self.out_fix.setText(f"Use “{self.session.root.name}{FMT_SUFFIX.get(f, '_out')}” inside it")
        self._check_out()
        self._split_changed()
        self._refresh_summary()

    def _nav_keys(self):
        return [k for k, _t, _i in SECTIONS]

    def _section_changed(self, i: int):
        if i < 0:
            return
        self.pages.setCurrentIndex(i)
        if SECTIONS[i][0] == "split":
            self._maybe_preview()

    def show_section(self, key: str):
        """Open the window on a step: output | classes | split | files | options (convert_yolo: YOLO, Output)."""
        if key == "convert_yolo":
            if not self.is_yolo() and not self.cls:
                self.fmt.setCurrentIndex(max(0, self.fmt.findData(
                    "yolo_seg" if self.survey is not None and self.survey.suggested_task() == "seg" else "yolo_det")))
            key = "output"
        if key in self._nav_keys():
            self.nav.setCurrentRow(self._nav_keys().index(key))

    def _maybe_preview(self):
        """The Split step on a dataset of a size that plans in seconds: plan it right away."""
        if (self.new.isChecked() and self.plan is None and self.worker is None
                and (self.survey is not None or self.cls) and len(self._samples()[0]) <= 30000):
            self.preview_split()

    def _refresh_nav(self, problems: dict):
        f = self.fmt_key()
        out = Path(self.out.text().strip() or ".").name
        n_cls = len(self.classes())
        v, t = self.val.value(), self.test.value()
        split = ("one set" if self.flat.isChecked() else "the dataset's splits" if self.keep.isChecked() else
                 f"new · {100 - v - t} / {v}" + (f" / {t}" if t else "") + (" · planned" if self.plan else ""))
        files = ("flattened, names made unique" if self.folders.currentData() == "flat" and self.folders.isEnabled()
                 else "sub-folders kept") + (" · links" if self.link.currentData() == "hardlink" else " · copies")
        lo = [w for w, c in (("flagged", self.ex_flag), ("broken", self.ex_broken)) if c.isChecked()]
        opts = ("leave out " + " & ".join(lo) if lo else "every image") + \
            ("" if self.bg.isChecked() else " · no background")
        subs = {"output": f"{SHORT.get(f, f)} → {out}",
                "classes": (f"{n_cls} of {self.list.count()} · ids in list order" if self.uses_classes()
                            else "the image flags"),
                "split": split, "files": files, "options": opts}
        for i in range(self.nav.count()):
            it = self.nav.item(i)
            key = it.data(ROLE_NAV_KEY)
            it.setData(ROLE_NAV_SUB, subs.get(key, ""))
            it.setData(ROLE_NAV_STATE, problems.get(key, ""))

    # ================================================================== classes
    def _counts(self):
        s = self.session
        objs = s.label_totals()
        imgs = Counter()
        for c in s.index.values():
            imgs.update(c.keys())
        return objs, imgs

    def _fill_classes(self):
        objs, imgs = self._counts()
        saved = self.saved.get("classes")               # [[name, ticked], …] from the last conversion
        outs = {}
        if saved:
            order = [(r[0], bool(r[1])) for r in saved]
            outs = {r[0]: r[2] for r in saved if len(r) > 2 and r[2]}
            have = {n for n, _t in order}
            order += [(n, True) for n in sorted(objs, key=str.lower) if n not in have]
        elif self.app.user_classes:                    # a loaded classes file gives the order
            order = [(n, True) for n in self.app.user_classes]
            order += [(n, True) for n in sorted(objs, key=str.lower) if n not in set(self.app.user_classes)]
        elif self._own_order():                        # a YOLO / COCO dataset: its own ids
            own = self._own_order()
            order = [(n, True) for n in own] + [(n, True) for n in sorted(objs, key=str.lower) if n not in set(own)]
        else:
            order = [(n, True) for n in sorted(objs, key=str.lower)]
        self.list.blockSignals(True)
        self.list.clear()
        for name, on in order:
            self._add_item(name, on, objs.get(name, 0), imgs.get(name, 0), outs.get(name, ""))
        self.list.blockSignals(False)
        self._renumber()

    def _own_order(self) -> list[str]:
        s = self.session
        if getattr(s.adapter, "mapping", None):
            return []
        try:
            base = s.base_adapter
            names = list(getattr(base, "class_names", None) or [])
        except Exception as exc:                      # noqa: BLE001 — no class list: alphabetical
            _ignored(exc)
            return []
        return [n for n in names if n] if (s.is_yolo or s.type == DatasetType.COCO) else []

    def _add_item(self, name, on, objs, imgs, out=""):
        it = QListWidgetItem(name)
        it.setData(ROLE_NAME, name)
        it.setData(ROLE_OUT, out or "")
        it.setData(ROLE_OBJ, objs)
        it.setData(ROLE_IMG, imgs)
        it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsDragEnabled)
        it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        if self.survey is not None:
            it.setData(ROLE_TYPES, dict(self.survey.types.get(name, {})))
        self.list.addItem(it)
        return it

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    @staticmethod
    def _out(it) -> str:
        return it.data(ROLE_OUT) or it.data(ROLE_NAME)

    def classes(self) -> list[str]:
        """The output classes, in id order (a merged class once, at its first ticked row)."""
        return list(dict.fromkeys(self._out(it) for it in self._items() if it.checkState() == Qt.CheckState.Checked))

    def rename_map(self) -> dict:
        return {it.data(ROLE_NAME): self._out(it) for it in self._items()
                if it.checkState() == Qt.CheckState.Checked and self._out(it) != it.data(ROLE_NAME)}

    def _renumber(self):
        self.list.blockSignals(True)
        ids = {c: i for i, c in enumerate(self.classes())}
        for it in self._items():
            it.setData(ROLE_ID, ids.get(self._out(it)) if it.checkState() == Qt.CheckState.Checked else None)
        self.list.blockSignals(False)
        self.list.viewport().update()
        n = self.list.count()
        merged = sum(1 for it in self._items() if it.checkState() == Qt.CheckState.Checked) - len(ids)
        self.count_lbl.setText(f"{len(ids)} of {n} classes included" + (f" · {merged} merged" if merged else ""))
        if hasattr(self, "parts_box"):
            self._refresh_part_choices()
        self._refresh_summary()

    # ------------------------------------------------------------------ write as / merge
    def _class_menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        from PyQt6.QtWidgets import QMenu
        m = QMenu(self)
        m.addAction("Write as…  (F2)", lambda: self._rename(it))
        into = m.addMenu("Merge into")
        for c in self.classes():
            if c != self._out(it):
                into.addAction(c, lambda c=c: self._set_out([it], c))
        into.setEnabled(not into.isEmpty())
        if self._out(it) != it.data(ROLE_NAME):
            m.addAction("Write under its own name", lambda: self._set_out([it], ""))
        sel = [x for x in self.list.selectedItems() if x is not it]
        if sel:
            m.addSeparator()
            m.addAction(f"Merge the {len(sel) + 1} selected into “{self._out(it)}”",
                        lambda: self._set_out(sel, self._out(it)))
        m.exec(self.list.viewport().mapToGlobal(pos))

    def _rename_selected(self):
        it = self.list.currentItem()
        if it is not None:
            self._rename(it)

    def _rename(self, it):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Write as", f"Write “{it.data(ROLE_NAME)}” as (an existing class name "
                                        "merges them):", text=self._out(it))
        if ok:
            self._set_out([it], name.strip())

    def _set_out(self, items, out: str):
        self.list.blockSignals(True)
        for it in items:
            it.setData(ROLE_OUT, "" if not out or out == it.data(ROLE_NAME) else out)
            if out:
                it.setCheckState(Qt.CheckState.Checked)
        self.list.blockSignals(False)
        self._renumber()

    def _move(self, step):
        rows = sorted({self.list.row(it) for it in self.list.selectedItems()}, reverse=step > 0)
        if not rows:
            return
        if (step < 0 and rows[-1 if step > 0 else 0] == 0) or (step > 0 and rows[0] == self.list.count() - 1):
            return
        self.list.blockSignals(True)
        for r in rows:
            it = self.list.takeItem(r)
            self.list.insertItem(r + step, it)
            it.setSelected(True)
        self.list.blockSignals(False)
        self._renumber()

    def _tick_all(self, on):
        self.list.blockSignals(True)
        for it in self._items():
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self._renumber()

    def _sort(self, _i=None):
        how = self.sort_box.currentData()
        self.sort_box.setCurrentIndex(0)
        if not how:
            return
        rows = [(it.data(ROLE_NAME), it.checkState() == Qt.CheckState.Checked, it.data(ROLE_OBJ) or 0,
                 it.data(ROLE_IMG) or 0, it.data(ROLE_OUT) or "") for it in self._items()]
        if how == "az":
            rows.sort(key=lambda r: r[0].lower())
        elif how == "count":
            rows.sort(key=lambda r: (-r[2], r[0].lower()))
        else:
            rows.sort(key=lambda r: not r[1])
        self.list.blockSignals(True)
        self.list.clear()
        for name, on, o, i, out in rows:
            self._add_item(name, on, o, i, out)
        self.list.blockSignals(False)
        self._renumber()

    def _load_order(self):
        from AIO_QC.core.class_map import load_classes_file
        start = str(self.session.root)
        path, _ = QFileDialog.getOpenFileName(self, "Class order from …", start,
                                              "Class lists (*.yaml *.yml *.txt *.names);;All files (*)")
        if not path:
            return
        try:
            names = [n for n in load_classes_file(path) if n]
        except Exception as exc:
            _ignored(exc)
            QMessageBox.warning(self, self.TITLE, f"Could not read {Path(path).name}:\n{exc}")
            return
        if not names:
            QMessageBox.warning(self, self.TITLE, f"No class names found in {Path(path).name}.")
            return
        self.apply_order(names, source=Path(path).name)

    def apply_order(self, names: list[str], source: str = "the file"):
        """The file's classes first, in its order and ticked (also those this dataset lacks: their ids stay the
        model's); the dataset's other classes after them, unticked."""
        names = list(dict.fromkeys(names))
        objs, imgs = self._counts()
        prev = {it.data(ROLE_NAME): it.data(ROLE_OUT) or "" for it in self._items()}
        self.list.blockSignals(True)
        self.list.clear()
        for n in names:
            self._add_item(n, True, objs.get(n, 0), imgs.get(n, 0))
        rest = [n for n in sorted(objs, key=str.lower) if n not in set(names)]
        for n in rest:                                 # a label written as one of the file's classes stays merged
            self._add_item(n, prev.get(n, "") in names, objs.get(n, 0), imgs.get(n, 0), prev.get(n, ""))
        rest = [n for n in rest if prev.get(n, "") not in names]
        self.list.blockSignals(False)
        self._renumber()
        absent = [n for n in names if not objs.get(n)]
        msg = f"Order from {source}: {len(names)} classes"
        if rest:
            msg += f" · {len(rest)} other label(s) of this dataset left out ({', '.join(rest[:4])}" + \
                   (" …" if len(rest) > 4 else "") + ")"
        if absent:
            msg += f" · {len(absent)} not in this dataset (kept, for their ids)"
        self.status.setText(msg)

    # ================================================================== more options / parts
    def _add_part_rule(self, part="", whole="", pct=80):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        pa, wh = QComboBox(), QComboBox()
        for c in (pa, wh):
            c.setMinimumContentsLength(10)
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        sp = QSpinBox()
        sp.setRange(10, 100)
        sp.setValue(int(pct))
        sp.setSuffix(" % inside")
        rm = utils.newButton("", icon="close-x", tip="Remove this rule")
        rm.setFixedWidth(30)
        h.addWidget(pa, 1)
        h.addWidget(QLabel("inside"))
        h.addWidget(wh, 1)
        h.addWidget(sp)
        h.addWidget(rm)
        rec = {"row": row, "part": pa, "whole": wh, "pct": sp, "want": (part, whole)}
        self._part_rows.append(rec)
        rm.clicked.connect(lambda: self._remove_part_rule(rec))
        self.parts_box.addWidget(row)
        self._refresh_part_choices()
        for c in (pa, wh):
            c.currentIndexChanged.connect(lambda _i: (self._split_changed(), self._refresh_summary()))
        sp.valueChanged.connect(lambda _v: self._refresh_summary())
        if not part and pa.count() > 1:
            pa.setCurrentIndex(pa.count() - 1)

    def _remove_part_rule(self, rec):
        self._part_rows.remove(rec)
        rec["row"].setParent(None)
        rec["row"].deleteLater()
        self._split_changed()
        self._refresh_summary()

    def _refresh_part_choices(self):
        classes = self.classes()
        for rec in getattr(self, "_part_rows", []):
            for key, i in (("part", 0), ("whole", 1)):
                c = rec[key]
                want = c.currentText() or rec["want"][i]
                c.blockSignals(True)
                c.clear()
                c.addItems(classes)
                k = c.findText(want)
                c.setCurrentIndex(k if k >= 0 else 0)
                c.blockSignals(False)

    def parts(self) -> list:
        """[(part, [wholes], share)] — rules for one part are joined (inside ANY of its wholes)."""
        by: dict = {}
        for rec in self._part_rows:
            p, w = rec["part"].currentText(), rec["whole"].currentText()
            if p and w and p != w:
                e = by.setdefault(p, [[], 1.0])
                e[0].append(w)
                e[1] = min(e[1], rec["pct"].value() / 100)
        return [(p, ws, r) for p, (ws, r) in by.items()]

    # ================================================================== split
    def _split_changed(self):
        """Settings changed: an earlier preview no longer applies."""
        if getattr(self, "plan", None) is not None and self.plan[0] != self._split_sig():
            self.plan = None
            self.preview_panel.setVisible(False)
        if hasattr(self, "train_lbl"):
            tr = 100 - self.val.value() - self.test.value()
            self.train_lbl.setText(f"→ {tr} % train" + ("" if tr >= 50 else " ⚠"))
        self._split_hint()
        if hasattr(self, "summary"):
            self._refresh_summary()

    def _split_hint(self):
        if not hasattr(self, "split_note"):
            return
        n = len(self._samples()[0]) if self.session is not None else 0
        v, t = round(n * self.val.value() / 100), round(n * self.test.value() / 100)
        msg = f"≈ {v:,} val" + (f" · {t:,} test" if t else "") + f" of {n:,} images."
        if n and v < 30:
            msg += f" <span style='color:{utils.hexc('warn')}'>Few val images: metrics will be noisy — a larger " \
                   "share (15–20 %) helps on a small dataset.</span>"
        if self.plan is not None:
            rep = self.plan[2].report
            notes = [r for r in rep["features"] if r["status"] not in ("ok", "train only (rare)")]
            msg = (f"Planned: {', '.join(f'{k} {v:,}' for k, v in rep['splits'].items())}"
                   f" · {rep['groups']['multi']:,} groups kept together · leaks {rep['groups']['split_across']}"
                   + (f" · {rep['repaired']} repair swap(s)" if rep.get("repaired") else "")
                   + (f" · {len(notes)} feature note(s) (in the report)" if notes else "") + ".")
        self.split_note.setText(msg)

    def _split_sig(self):
        return (self.val.value(), self.test.value(), self.seed.value(), self.rare.value(), self.g_dups.isChecked(),
                self.g_video.isChecked(), self.g_look.isChecked(), self.g_ident.isChecked(), tuple(self.classes()),
                tuple(sorted(self.rename_map().items())), self.fold.isChecked(), tuple(map(str, self.parts())),
                self.uses_classes(), self._samples_digest())

    def _idents(self) -> list:
        """[(key, identity)] from this session's OCR Check reads (confident, 4+ characters)."""
        if not (self.g_ident.isChecked() and self.g_ident.isEnabled()):
            return []
        from AIO_QC.core.validators import identity_key
        ocr = self.app.open_dialogs.get("ocr_check")
        reads = ocr.ctx.get("reads", []) if ocr is not None and getattr(ocr, "ctx", None) else []
        out = []
        for r in reads:
            txt = (r.res or {}).get("text") or ""
            if float((r.res or {}).get("conf") or 0) >= 0.6 and len(txt) >= 4:
                out.append((r.key, identity_key(txt)))
        return out

    def _split_fn(self):
        """-> a function for a worker: (signature, items, plan)."""
        sig = self._split_sig()
        samples = self._samples()[0]
        s = self.session
        ad, cache = s.adapter, s.cache
        ids = YC.LabelIds(self.classes(), self.rename_map(), self.fold.isChecked())
        part_classes = {p for p, _w, _r in self.parts()}
        focus = set(self.classes()) - part_classes if part_classes else None
        val, test, seed, rare = self.val.value() / 100, self.test.value() / 100, self.seed.value(), self.rare.value()
        dups, video, look = self.g_dups.isChecked(), self.g_video.isChecked(), self.g_look.isChecked()
        cancel, bus = self.cancel, self.bus
        sv = self.survey
        idents = self._idents()
        if self.cls:                                   # a classifier: its classes are the image's tags
            view = EX.ClassView(ad, self.classes(), self.rename_map(), self.fold.isChecked(), tags=True) \
                if self.uses_classes() else ad

            def work_cls():
                from AIO_QC.core.split import build_items, plan_split
                items = build_items(samples, view, cache, classification=True, dups=dups, near=dups, video=video,
                                    lighting=look, measure=look, idents=idents, cancel=cancel,
                                    progress=lambda i, n: bus.step.emit(i, n, "split"))
                return sig, items, plan_split(items, val, test, seed, rare)
            return work_cls

        def work():
            from AIO_QC.core.split import build_items, plan_split
            survey = sv or YC.survey(samples, ad, cancel=cancel)
            per = {}
            for x in samples:                          # written classes only, under their output names
                labs, boxes = survey.per_sample.get(x.key, (Counter(), []))
                L = Counter()
                for k, n in labs.items():
                    hit = ids.get(k)
                    if hit is not None:
                        L[hit[1]] += n
                B = []
                for b in boxes:
                    hit = ids.get(b[4]) if len(b) > 4 else None
                    if hit is not None:
                        B.append(b[:4] + (hit[1],))
                per[x.key] = (L, B)
            items = build_items(samples, ad, cache, per_sample=per, dups=dups, near=dups, video=video,
                                lighting=look, measure=look, focus=focus, idents=idents, cancel=cancel,
                                progress=lambda i, n: bus.step.emit(i, n, "split"))
            return sig, items, plan_split(items, val, test, seed, rare)
        return work

    def preview_split(self):
        if self.worker is not None:
            return
        self.new.setChecked(True)
        self.cancel.clear()
        self._busy(True, "Planning the split …")
        w = FuncWorker(self._split_fn(), self)
        w.done.connect(self._split_done)
        w.failed.connect(lambda m: (self._busy(False), self.status.setText(f"⚠ Split failed: {m}")))
        w.finished.connect(w.deleteLater)
        self.worker = w
        w.start()

    def _split_done(self, res):
        self._busy(False)
        self.status.setText("")
        self._show_plan(res)

    def _show_plan(self, res):
        self.plan = res
        rep = res[2].report
        self.split_bar.set_counts(rep["splits"])
        target = {"val": self.val.value() / 100, "test": self.test.value() / 100}
        target["train"] = 1 - target["val"] - target["test"]
        self.class_bars.set_data(rep["classes"], target)
        rows = sorted(rep["features"], key=lambda r: (r["status"] == "ok", r["kind"], -r["images"]))[:5000]
        self.ftable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for c, v in enumerate((r["kind"], r["value"], r["images"], r["train"], r["val"], r["test"], r["status"])):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole, v)
                if c == 6:
                    it.setForeground(utils.theme_color("ok" if v == "ok" else "muted" if "rare" in v and "IN" not in v
                                                       else "warn" if v.startswith("missing") else "error"))
                self.ftable.setItem(i, c, it)
        self.preview_panel.setVisible(True)
        self._split_hint()
        self._refresh_summary()

    # ================================================================== survey
    def _start_survey(self):
        s = self.session
        if self.cls:                                   # a classifier has no shapes to look at
            self.survey = YC.Survey()
            return
        samples = list(s.samples)
        ad = s.adapter
        cancel = self.cancel

        def work():
            return YC.survey(samples, ad, progress=lambda i, n: self.bus.step.emit(i, n, "survey"), cancel=cancel)
        w = FuncWorker(work, self)
        w.done.connect(self._surveyed)
        w.failed.connect(lambda m: self.task_hint.setText(f"Could not look at the shapes: {m}"))
        w.finished.connect(w.deleteLater)
        self._survey_worker = w
        w.start()

    def _surveyed(self, sv):
        if self.cancel.is_set() and self.worker is None:
            return
        self.survey = sv
        for it in self._items():
            it.setData(ROLE_TYPES, dict(sv.types.get(it.data(ROLE_NAME), {})))
        t = Counter()
        for c in sv.types.values():
            t.update(c)
        words = Counter()
        for k, n in t.items():
            words[TYPE_WORD.get(k, k)] += n
        sug = sv.suggested_task()
        txt = " · ".join(_n(n, k) for k, n in words.most_common()) or "no shapes"
        self.task_hint.setText(f"This dataset has {txt}. "
                               f"<b>For YOLO: {'segmentation' if sug == 'seg' else 'detection'}</b>"
                               + (" (most shapes are outlines)." if sug == "seg" else " (most shapes are boxes)."))
        if "task" not in self.saved and "format" not in self.saved and self.is_yolo():
            self.fmt.setCurrentIndex(max(0, self.fmt.findData("yolo_seg" if sug == "seg" else "yolo_det")))
        if self.status.text().startswith("Looking"):
            self.status.setText("")
        self.list.viewport().update()
        self._refresh_summary()
        if self.nav.currentRow() == self._nav_keys().index("split"):
            self._maybe_preview()

    # ================================================================== save to
    def _browse(self):
        from AIO_QC.utils.save_dirs import ask_dir
        d = ask_dir(self, "Save the dataset to", kind="datasets", current=self.out.text())
        if d:
            self.out.setText(d)

    def out_problem(self) -> tuple[str, str]:
        """-> (level ok | replace | warn | error, html)."""
        txt = self.out.text().strip()
        if not txt:
            return "error", "Choose a folder."
        out = Path(txt).expanduser()
        root = self.session.root.resolve()
        try:
            o = out.resolve()
        except OSError:
            return "error", "That is not a usable folder."
        if o == root or root in o.parents:
            return "error", "That folder is inside the dataset — choose one outside it."
        if o in root.parents:
            return "error", "That folder contains the dataset — choose an empty or new folder."
        if not out.exists():
            return "ok", "A new folder — it will be created."
        if not out.is_dir():
            return "error", "That is a file, not a folder."
        prev = YC.previous_output(out)
        if prev is not None:
            task = prev.get("format") or prev.get("task", "")
            what = SHORT.get(task) or ("YOLO " + YC.TASKS.get(task, "").split(" (")[0].lower())
            return "replace", (f"Holds an earlier conversion ({what}, {len(prev.get('classes', []))} classes, "
                               f"{prev.get('time', '')[:16]}) — it will be <b>replaced</b>; other files there stay.")
        try:
            empty = not any(x for x in out.iterdir() if x.name != ".aio_qc_provenance.json")
        except OSError:
            return "error", "That folder cannot be read."
        if empty:
            return "ok", "An empty folder."
        return "warn", "That folder is not empty — choose an empty or new folder."

    def _check_out(self):
        level, html = self.out_problem()
        color = {"ok": "ok", "replace": "warn", "warn": "error", "error": "error"}[level]
        self.out_state.setText(f"<span style='color:{utils.hexc(color)}'>{html}</span>")
        self.out_fix.setVisible(level == "warn")
        self.go.setEnabled(level in ("ok", "replace") and self.worker is None)

    def _use_subfolder(self):
        base = Path(self.out.text().strip())
        name = f"{self.session.root.name}{FMT_SUFFIX.get(self.fmt_key(), '_out')}"
        cand = base / name
        k = 2
        while cand.exists() and YC.previous_output(cand) is None and any(cand.iterdir()):
            cand = base / f"{name}_{k}"
            k += 1
        self.out.setText(str(cand))

    # ================================================================== summary
    def _forget_samples(self):
        self._samples_memo = None

    def _samples(self):
        """(the images written, how many are left out) — flagged / broken ones out as chosen. Remembered for a few
        seconds: the summary, the split check and the naming all ask, and a dataset can have a million images."""
        key = (self.ex_flag.isChecked(), self.ex_broken.isChecked(), len(self.session.samples))
        m = self._samples_memo
        if m is not None and m[0] == key and time.monotonic() - m[1] < 3.0:
            return m[2]
        res = self._samples_now()
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for x in res[0]:
            h.update(str(x.key).encode("utf-8", "surrogatepass"))
            h.update(b"\0")
        self._samples_memo = (key, time.monotonic(), res, h.hexdigest())
        return res

    def _samples_digest(self) -> str:
        self._samples()
        return self._samples_memo[3]

    def _samples_now(self):
        s = self.session
        broken = set()
        if self.ex_broken.isChecked():
            for tool in ("integrity", "image_quality"):
                for k, lst in s.issues.by_tool.get(tool, {}).items():
                    if any(i.severity == "error" and not i.text.startswith(("EXIF rotation mismatch", "size mismatch"))
                           for i in lst):
                        broken.add(k)
        keep = []
        for x in s.samples:
            if self.ex_flag.isChecked() and s.review.status(x.key) == REVIEW_FLAG:
                continue
            if x.key in broken:
                continue
            keep.append(x)
        return keep, len(s.samples) - len(keep)

    def _refresh_summary(self):
        self._summary_timer.start()

    def _summary_now(self):
        s = self.session
        if s is None:
            return
        f, y, uses = self.fmt_key(), self.is_yolo(), self.uses_classes()
        out_classes = self.classes() if uses else []
        objs, _imgs = self._counts()
        ids = YC.LabelIds(out_classes, self.rename_map(), self.fold.isChecked())
        inc = {lab for lab in objs if ids.get(lab) is not None} if uses else set(objs)   # dataset labels written
        samples, left_out = self._samples()
        bg = 0
        for x in samples:
            if inc.isdisjoint(s.index.get(x.key, ())):
                bg += 1
        n_obj = sum(objs.get(c, 0) for c in inc)
        out_lab = {k: v for k, v in objs.items() if k not in inc and v}
        n_img = len(samples) - (0 if self.bg.isChecked() else bg)
        sv = self.survey
        if sv is not None and y:
            n_obj -= sum(sv.types.get(lab, {}).get(k, 0) for lab in inc for k in YC.NOT_AREA)
        approx = "" if sv is not None and not left_out else "about "
        what = "images" if self.cls else "objects"
        L = [f"<b>{_n(n_img, 'images')}</b>" + ("" if self.cls else f" · {approx}<b>{_n(n_obj, what)}</b>")
             + (f" in <b>{_n(len(out_classes), 'classes')}</b>" if uses else "")
             + f" as <b>{SHORT.get(f, f)}</b>"
             + (f" · {_n(bg, 'background images')}" if bg and self.bg.isChecked() and not self.cls else "") + "."]
        if self.new.isChecked():
            if self.plan is not None:
                L.append("Split: " + " · ".join(f"{k} {v:,}" for k, v in self.plan[2].report["splits"].items()))
            else:
                v, t_ = self.val.value(), self.test.value()
                L.append(f"Split: a new one, {100 - v - t_} / {v}" + (f" / {t_}" if t_ else "") + " % (planned when "
                         "you convert — or see it first under Split)")
        elif self.keep.isChecked():
            norm = {"valid": "val", "validation": "val"}
            c = Counter(norm.get(x.split, x.split) or "train" for x in samples)
            L.append("Split: as in the dataset — " + " · ".join(f"{k} {v:,}" for k, v in c.items()))
        notes, problems = [], {}
        if uses and not out_classes:
            notes.append(f"<span style='color:{utils.hexc('error')}'>Tick at least one class.</span>")
            problems["classes"] = "error"
        ren = {k: v for k, v in ((lab, ids.get(lab)) for lab in inc) if v is not None and v[1] != k} if uses else {}
        if ren:
            ex = ", ".join(f"{k} → {v[1]}" for k, v in list(ren.items())[:3])
            notes.append(f"{_n(len(ren), 'labels')} written under another name ({ex}" + (" …" if len(ren) > 3 else "")
                         + ")")
        variants = Counter(YC._norm_label(k) for k in objs)
        twins = [k for k, n in variants.items() if n > 1]
        if twins and not self.fold.isChecked() and uses:
            notes.append(f"<span style='color:{utils.hexc('warn')}'>{_n(len(twins), 'labels')} differ only in case or "
                         "spaces (e.g. " + ", ".join(f"“{k}”" for k in objs if YC._norm_label(k) == twins[0])
                         + ") — <i>Options ▸ More options ▸ Merge labels that differ only in case</i></span>")
            problems.setdefault("classes", "warn")
        if self.left_out.currentData() == "skip" and out_lab:
            notes.append("images with a class that is not written are left out")
        if y:
            for p_, w_, r_ in self.parts():
                notes.append(f"{p_} written only inside {' / '.join(w_)} (≥ {100 * r_:.0f} %)")
        if out_lab:
            top = ", ".join(f"{k} ({v:,})" for k, v in sorted(out_lab.items(), key=lambda kv: -kv[1])[:4])
            notes.append(f"Not written: {sum(out_lab.values()):,} {what} — {top}" + (" …" if len(out_lab) > 4 else ""))
        if bg and not self.bg.isChecked():
            notes.append(f"{bg:,} image(s) without a written class are left out")
        if left_out:
            notes.append(f"{left_out:,} flagged / broken image(s) left out")
        if sv is not None and not self.cls:
            t = Counter()
            for lab in inc:
                t.update(sv.types.get(lab, {}))
            na = sum(t[k] for k in YC.NOT_AREA)
            if y and na:
                notes.append(f"{_n(na, 'lines / points')} can't become {'boxes' if f == 'yolo_det' else 'polygons'}"
                             " — not written")
            if y and sv.groups and self.merge.isChecked():
                notes.append(f"{sv.groups:,} object(s) drawn in pieces will be joined")
            if f == "yolo_det" and (t["polygon"] or t["mask"]):
                notes.append("outlines become their bounding boxes")
            if f == "yolo_seg" and (t["rectangle"] or t["oriented rectangle"]):
                notes.append(f"{t['rectangle'] + t['oriented rectangle']:,} boxes become 4-corner polygons")
            if f in ("voc", "yolo_obb", "dota") and (t["polygon"] or t["mask"] or t["circle"]):
                notes.append("outlines become their " + ("bounding boxes" if f == "voc" else "tightest rotated boxes"))
            if sv.missing_image:
                notes.append(f"<span style='color:{utils.hexc('warn')}'>{len(sv.missing_image):,} image(s) not found "
                             "— skipped</span>")
        elif sv is None:
            notes.append("<i>still looking at the shapes …</i>")
        if notes:
            L.append("<br>".join("· " + n for n in notes))
        naming_ok = self.naming_ready()
        if not naming_ok:
            L.append(f"<span style='color:{utils.hexc('warn')}'>Flatten: names still clash — Number the rest, or "
                     "change the naming pattern</span>")
            problems["files"] = "error"
        level = self.out_problem()[0]
        if level in ("error", "warn"):
            problems["output"] = "error"
        elif level == "replace":
            problems["output"] = "warn"
        if self.new.isChecked() and self.plan is not None:
            rep = self.plan[2].report
            if rep["groups"]["split_across"] or any(r["status"] in ("MISSING IN TRAIN", "RARE IN VAL/TEST")
                                                    for r in rep["features"]):
                problems["split"] = "warn"
        self.summary.setText("<br><br>".join(L[:1] + ["<br>".join(L[1:])] if len(L) > 1 else L))
        self._refresh_nav(problems)
        self.go.setEnabled(bool(inc or not uses) and n_img > 0 and level in ("ok", "replace") and self.worker is None
                           and naming_ok)

    def _name_by(self) -> str:
        return "annotation" if self.session.type in (DatasetType.LABELME, DatasetType.CLS_LABELME) else "image"

    def _flat_files(self) -> bool:
        return self.folders.currentData() == "flat" and self.folders.isEnabled()

    def _folders_changed(self):
        flat = self._flat_files()
        self.naming.setVisible(flat)
        if flat:
            by = self._by_split(self._samples()[0], plan=None)
            items, _by = YC.naming_items(by, self.session.root, self._name_by())
            out = str(Path(self.out.text().strip() or ".").expanduser().resolve())
            self.naming.set_items(items, flat=True, root_len=len(out) + len("/train/images"))
        self._refresh_summary()

    def _by_split(self, samples, plan=None) -> dict:
        if self.flat.isChecked() or (self.new.isChecked() and plan is None):
            return {"": samples}                    # (a new split is not known yet: names unique across all)
        if self.keep.isChecked():
            norm = {"valid": "val", "validation": "val"}
            by = {}
            for x in samples:
                by.setdefault(norm.get(x.split, x.split) or "train", []).append(x)
            return by
        by = {}
        for x in samples:
            by.setdefault(plan[2].assign.get(x.key, "train"), []).append(x)
        return by

    def naming_ready(self) -> bool:
        return not self._flat_files() or self.naming.is_ready()

    # ================================================================== convert
    def options(self) -> YC.Options:
        return YC.Options(task=self._yolo_task(), classes=self.classes(),
                          layout="flat" if self.flat.isChecked() else "splits",
                          link=self.link.currentData(), background="keep" if self.bg.isChecked() else "skip",
                          merge_groups=self.merge.isChecked(), rename=self.rename_map(),
                          fold_case=self.fold.isChecked(), min_area=float(self.min_area.value()),
                          left_out=self.left_out.currentData(), broken=self.broken_box.currentData(),
                          parts=self.parts(), orphans=self.orphans.currentData(),
                          fix_images=self.fix_img.isChecked(), verify=self.verify_box.isChecked(),
                          folders="flat" if self._flat_files() else "keep",
                          naming=self.naming.scheme().to_dict() if self._flat_files() else {}, name_by=self._name_by())

    def _remember(self):
        d = {"format": self.fmt_key(), "out": self.out.text().strip(),
             "crop": {"pad": self.crop_pad.value(), "square": self.crop_square.isChecked(), "min": self.crop_min.value()},
             "layout": "flat" if self.flat.isChecked() else "splits" if self.keep.isChecked() else "new",
             "val": self.val.value(), "test": self.test.value(), "merge": self.merge.isChecked(),
             "background": self.bg.isChecked(), "ex_flag": self.ex_flag.isChecked(),
             "ex_broken": self.ex_broken.isChecked(), "link": self.link.currentData(),
             "seed": self.seed.value(), "rare": self.rare.value(), "dups": self.g_dups.isChecked(),
             "video": self.g_video.isChecked(), "measure": self.g_look.isChecked(), "fold": self.fold.isChecked(),
             "min_area": self.min_area.value(), "left_out": self.left_out.currentData(),
             "broken": self.broken_box.currentData(), "fix_images": self.fix_img.isChecked(),
             "verify": self.verify_box.isChecked(), "orphans": self.orphans.currentData(),
             "folders": self.folders.currentData(), "ident": self.g_ident.isChecked(),
             "parts": [[r["part"].currentText(), r["whole"].currentText(), r["pct"].value()] for r in self._part_rows],
             "classes": [[it.data(ROLE_NAME), it.checkState() == Qt.CheckState.Checked, it.data(ROLE_OUT) or ""]
                         for it in self._items()]}
        utils.app_settings().setValue(self._key, json.dumps(d, ensure_ascii=False))

    def start(self):
        if self.worker is not None:
            return
        level, html = self.out_problem()
        if level not in ("ok", "replace"):
            self._check_out()
            self.nav.setCurrentRow(0)
            return
        opts = self.options()
        if self.uses_classes() and not opts.classes:
            self.nav.setCurrentRow(1)
            return
        if not self.naming_ready():
            self.naming.wait()
            if not self.naming_ready():
                self.status.setText("⚠ Flatten: some names still clash — Number the rest, or change the pattern")
                self.nav.setCurrentRow(self._nav_keys().index("files"))
                return
        out = Path(self.out.text().strip()).expanduser()
        if level == "replace":
            YC.remove_previous(out)
        self._remember()
        s = self.session
        samples, _left = self._samples()
        layout = "flat" if self.flat.isChecked() else "keep" if self.keep.isChecked() else "new"
        ad, root = s.adapter, s.root
        planned = self.plan if (self.plan is not None and self.plan[0] == self._split_sig()) else None
        split_fn = self._split_fn() if layout == "new" and planned is None else None
        fmt = self.fmt_key()
        settings = {"format": fmt, "layout": layout, "val": self.val.value(), "test": self.test.value(),
                    "seed": self.seed.value(), "rare": self.rare.value(), "dups": self.g_dups.isChecked(),
                    "video": self.g_video.isChecked(), "look": self.g_look.isChecked(), "classes": opts.classes,
                    "rename": opts.rename, "parts": opts.parts}
        self.cancel.clear()
        cancel, bus = self.cancel, self.bus

        def by_split():
            plan = None
            if layout == "flat":
                by = {"": samples}
            elif layout == "keep":
                norm = {"valid": "val", "validation": "val"}
                by = {"train": [], "val": [], "test": []}
                for x in samples:
                    by.setdefault(norm.get(x.split, x.split) or "train", []).append(x)
            else:
                plan = planned or split_fn()
                by = {"train": [], "val": [], "test": []}
                assign = plan[2].assign
                for x in samples:
                    by[assign.get(x.key, "train")].append(x)
            return {k: v for k, v in by.items() if v}, plan

        if self.is_yolo():
            def work():
                by, plan = by_split()
                st = YC.convert(by, ad, root, out, opts, cancel=cancel, source_name=root.name,
                                progress=lambda i, n: bus.step.emit(i, n, "convert"))
                if plan is not None and not st.get("cancelled"):
                    from AIO_QC.core.split import write_reports
                    _sig, items, p_ = plan
                    write_reports(out / "reports", items, p_.assign, p_.report, settings,
                                  ["CONVERSION", "-" * 64, f"see {out.name}/conversion_report.txt"])
                    YC.add_created(out, "reports")
                return st, plan
            done = lambda res: self._done(res[0], out, opts, res[1])          # noqa: E731
        else:
            uses = self.uses_classes()
            names = list(opts.classes) if uses else sorted(self._counts()[0], key=str.lower)
            view = EX.ClassView(ad, opts.classes, opts.rename, opts.fold_case, opts.left_out,
                                "keep" if self.bg.isChecked() else "skip", tags=self.cls) if uses else ad
            naming = self.naming.scheme().to_dict() if self._flat_files() else None
            mode = self.link.currentData()
            crop = {"pad": self.crop_pad.value() / 100, "square": self.crop_square.isChecked(),
                    "min": self.crop_min.value()}
            is_cls = self.cls

            def work():
                import os
                out.mkdir(parents=True, exist_ok=True)
                before = set(os.listdir(out))
                by, plan = by_split()
                st = EX.export(by, view, root, out, fmt, names, mode, cancel=cancel, is_classification=is_cls,
                               naming=naming, crop=crop, progress=lambda i, n: bus.step.emit(i, n, "convert"))
                if not st.get("cancelled"):
                    from AIO_QC.core.io import atomic_write_text
                    if plan is not None:
                        _sig, items, p_ = plan
                        EX.write_reports(out, p_.report, p_.assign, settings, st, items)
                    atomic_write_text(out / "export_report.txt", EX.report_text(st, fmt, names, root.name))
                created = sorted(set(os.listdir(out)) - before - {YC.MANIFEST})
                EX.write_manifest(out, fmt, names, root.name, created)
                return st, plan
            done = lambda res: self._done_export(res[0], out, fmt, names, res[1])  # noqa: E731

        self._busy(True)
        self._t0 = time.time()
        w = FuncWorker(work, self)
        w.done.connect(done)
        w.failed.connect(lambda m: (self._busy(False), self.status.setText(f"⚠ Conversion failed: {m}")))
        w.finished.connect(w.deleteLater)
        self.worker = w
        w.start()

    def _busy(self, on, text="Converting …"):
        from AIO_QC.widgets import tasks
        self.progress.setVisible(on)
        self.stop_btn.setVisible(on)
        self.go.setEnabled(not on)
        self.preview_btn.setEnabled(not on)
        for b in (self.open_btn, self.reopen_btn, self.train_btn, self.undo_btn):
            b.setVisible(False if on else b.isVisible())
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        if on:
            self.progress.setRange(0, 0)
            self.status.setText(text)
            self._task = tasks.start(f"{self.TITLE} · {self.session.root.name}", cancel=self.cancel.set)
        else:
            self.worker = None
            self._check_out()

    def _on_step(self, i, n, what):
        if what == "survey":
            if self.survey is None and self.worker is None and n > 2000:
                self.status.setText(f"Looking at the shapes … {i:,} / {n:,}")
            return
        if self.worker is None:
            return
        if what == "split":
            self.progress.setRange(0, max(n, 1))
            self.progress.setValue(i)
            self.status.setText(f"Planning the split … {i:,} / {n:,}")
            return
        self.progress.setRange(0, max(n, 1))
        self.progress.setValue(i)
        if getattr(self, "_task", None) is not None:
            self._task.update(i / n if n else None, f"{i:,} / {n:,}")
        el = time.time() - self._t0
        left = el / max(i, 1) * (n - i) if i else 0
        self.status.setText(f"Converting {i:,} / {n:,}" + (f" · about {_dur(left)} left" if i > 50 and left > 2 else ""))

    def _done(self, st, out, opts, plan=None):
        self._busy(False)
        self._last_out, self._last_fmt = out, self.fmt_key()
        if st.get("cancelled"):
            self.status.setText("⏹ Stopped — the folder holds a partial conversion (Convert again replaces it, Undo "
                                "removes it).")
            self.undo_btn.setVisible(True)
            return
        if plan is not None:
            self._show_plan(plan)
        self._last_out = out
        n = sum(st["images"].values())
        nobj = sum(st["objects"].values())
        msg = (f"Converted {n:,} images · {nobj:,} objects · {len(opts.classes)} classes"
               + (f" · {len(st['skipped']):,} skipped" if st["skipped"] else ""))
        # read the dataset.yaml back: the names a trainer will see are the ones chosen
        from AIO_QC.core.class_map import parse_yaml_class_map
        back = parse_yaml_class_map(out / "dataset.yaml").names
        if back != opts.classes:
            msg += " · ⚠ dataset.yaml did not read back as written"
        v = st.get("verify")
        if v is not None:
            msg += " · verified" if (not v["problems"] and v["objects"] == +st["objects"]) else \
                f" · ⚠ {len(v['problems'])} label problem(s) (see the report)"
        lo = sum(st.get("left_out_images", {}).values())
        if lo:
            msg += f" · {lo:,} left out by your rules"
        self.status.setText(msg)
        for b in (self.open_btn, self.reopen_btn, self.train_btn, self.undo_btn):
            b.setVisible(True)
        details = (out / "conversion_report.txt").read_text(encoding="utf-8")
        if (out / "reports" / "split_report.txt").is_file():
            details += "\n" + (out / "reports" / "split_report.txt").read_text(encoding="utf-8")
        self.app.toast.show_message(f"<b>Converted to YOLO</b><br>{msg}", kind="success" if not st["skipped"] else "warn",
                                    actions=[("Open folder", self._open_folder)], details=details, key="yolo-convert")

    def _done_export(self, st, out, fmt, names, plan=None):
        self._busy(False)
        self._last_out, self._last_fmt = out, fmt
        if st.get("cancelled"):
            self.status.setText("⏹ Stopped — the folder holds a partial conversion (Convert again replaces it, Undo "
                                "removes it).")
            self.undo_btn.setVisible(True)
            return
        if plan is not None:
            self._show_plan(plan)
        n = sum(st["images"].values())
        lo = sum(st.get("left_out_images", {}).values())
        msg = (f"Converted {n:,} images to {SHORT.get(fmt, fmt)}"
               + (" (" + ", ".join(f"{k} {v:,}" for k, v in st["images"].items() if k) + ")"
                  if any(st["images"]) else "")
               + (f" · {sum(st['objects'].values()):,} objects" if st["objects"] and not self.cls else "")
               + (f" · {lo:,} left out by your rules" if lo else "")
               + (f" · {len(st['skipped']):,} skipped" if st["skipped"] else ""))
        self.status.setText(msg)
        self.open_btn.setVisible(True)
        self.undo_btn.setVisible(True)
        self.reopen_btn.setVisible(fmt in OPENABLE)
        self.train_btn.setVisible(fmt in ("cls_folders", "crops", "flags_folders"))      # a classifier, from folders
        details = (out / "export_report.txt").read_text(encoding="utf-8") \
            if (out / "export_report.txt").is_file() else msg
        self.app.toast.show_message(f"<b>Convert & Split finished</b><br>{msg}",
                                    kind="success" if not st["skipped"] else "warn",
                                    actions=[("Open folder", self._open_folder)], details=details, key="yolo-convert")

    def _undo(self):
        out = self._last_out
        if out is None:
            return
        if QMessageBox.question(self, self.TITLE, f"Remove what this conversion wrote into {out}?\n\nOnly its own "
                                "files and folders go; anything else in the folder stays.") \
                != QMessageBox.StandardButton.Yes:
            return
        n = YC.remove_previous(out)                       # (its provenance record goes with it)
        try:
            out.rmdir()                                    # only when nothing else is left in it
        except OSError:
            pass
        self.undo_btn.setVisible(False)
        for b in (self.open_btn, self.reopen_btn, self.train_btn):
            b.setVisible(False)
        self.status.setText(f"↶ Undone — {n} item(s) removed from {out}.")
        self._check_out()

    def _train(self):
        if self._last_out is None:
            return
        out = Path(self._last_out)
        self.app.open_training(out / "dataset.yaml" if (out / "dataset.yaml").is_file() else out)

    def _open_folder(self):
        if self._last_out is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_out)))

    def _open_in_app(self):
        out = self._last_out
        if out is None:
            return
        if QMessageBox.question(self, self.TITLE, f"Open {out.name} in AIO QC? This dataset closes.") \
                != QMessageBox.StandardButton.Yes:
            return
        app = self.app
        self.close()
        app.open_dataset(Path(out))

    def closeEvent(self, e):
        self.cancel.set()
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        for w in (self.worker, getattr(self, "_survey_worker", None)):
            try:
                if w is not None and w.isRunning():
                    w.wait(5000)
            except RuntimeError:
                pass
        self.app.forget_dialog(self)
        super().closeEvent(e)


def _dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


YoloConvertDialog = ConvertSplitDialog          # the name it had before Convert & Split was one window
