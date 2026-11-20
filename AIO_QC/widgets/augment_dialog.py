# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image Augmentation (Models tab, Ctrl+Alt+A) — build a recipe of effects and see it on your own images while you
change it; then either train with it (Train Model applies it in memory: no files) or write an augmented copy of the
dataset.

Left: the recipe (a saved one or a preset), how many effects an image gets, overall strength, and every effect by
family — tick it, set its chance and its range; the eye shows that effect alone. Middle: the original and five
variations of the shown image with their labels, redrawn as you change things (Shuffle: other random draws). Bottom:
Train with it · Generate files (copies per image, which images, rare classes boosted, originals included, where to).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                             QMessageBox, QProgressBar, QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget,
                             QToolButton, QVBoxLayout, QWidget)

from AIO_QC import utils
from AIO_QC.core import augment as A
from AIO_QC.core import augment_ops as G
from AIO_QC.core.dataset import DatasetType

from .inference_service import FuncWorker
from AIO_QC.core.diag import ignored as _ignored

PREVIEW_SIDE = 900            # the shown image is augmented at up to this size (fast, and close to what a model sees)
N_VARIATIONS = 5


def _fmt_of(session) -> str | None:
    if session is None:
        return None
    t = session.type
    if t in (DatasetType.YOLO, DatasetType.YOLO_SPLIT):
        return "yolo"
    if t == DatasetType.LABELME:
        return "labelme"
    if t == DatasetType.CLS_FOLDERS:
        return "images"
    return None


def test_card(w: int = 640, h: int = 480):
    """A picture to preview on when no dataset is open: colour bars, a gradient, text and two 'objects'."""
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    cols = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 0, 0)]
    for i, c in enumerate(cols):
        img[:h // 3, i * w // 7:(i + 1) * w // 7] = c
    img[h // 3:, :] = np.linspace(40, 215, w, dtype=np.uint8)[None, :, None]
    cv2.rectangle(img, (int(w * .12), int(h * .45)), (int(w * .42), int(h * .85)), (40, 90, 200), -1)
    cv2.circle(img, (int(w * .72), int(h * .66)), int(h * .16), (60, 170, 60), -1)
    cv2.putText(img, "AIO QC 123", (int(w * .14), int(h * .68)), cv2.FONT_HERSHEY_SIMPLEX, w / 640 * 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    shapes = [{"label": "box", "shape_type": "rectangle", "points": [[w * .12, h * .45], [w * .42, h * .85]]},
              {"label": "disc", "shape_type": "polygon",
               "points": [[w * .72 + h * .16 * np.cos(a), h * .66 + h * .16 * np.sin(a)]
                          for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]}]
    return img, shapes


def draw_shapes(img, shapes, colors: dict):
    import cv2
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else np.ascontiguousarray(img[..., :3]).copy()
    t = max(1, int(round(max(out.shape[:2]) / 400)))
    for s in shapes or []:
        P = np.asarray(s.get("points") or [], np.float64).reshape(-1, 2)
        if not len(P):
            continue
        c = colors.get(str(s.get("label", s.get("cls", ""))), (60, 220, 60))
        st = s.get("shape_type") or "polygon"
        if st == "rectangle" and len(P) >= 2:
            cv2.rectangle(out, tuple(int(v) for v in P[0]), tuple(int(v) for v in P[1]), c, t, cv2.LINE_AA)
        elif st == "circle" and len(P) >= 2:
            cv2.circle(out, tuple(int(v) for v in P[0]), int(np.hypot(*(P[1] - P[0]))), c, t, cv2.LINE_AA)
        elif st == "point":
            cv2.circle(out, tuple(int(v) for v in P[0]), 3 * t, c, -1, cv2.LINE_AA)
        else:
            cv2.polylines(out, [P.astype(np.int32)], st not in ("line", "linestrip"), c, t, cv2.LINE_AA)
    return out


def _pixmap(bgr) -> QPixmap:
    import cv2
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


# ====================================================================================================== effect rows
class OpRow(QFrame):
    changed = pyqtSignal()
    solo = pyqtSignal(str, bool)

    def __init__(self, op: dict):
        super().__init__()
        self.op = op
        self.key = op["key"]
        self.spins: dict = {}
        g = QGridLayout(self)
        g.setContentsMargins(6, 3, 6, 3)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(2)
        self.on = QCheckBox(op["title"])
        self.on.setToolTip(op["tip"])
        self.on.toggled.connect(self._toggled)
        self.chance = QSpinBox()
        self.chance.setRange(1, 100)
        self.chance.setSuffix(" %")
        self.chance.setValue(int(round((op["p"] or 0.3) * 100)))
        self.chance.setFixedWidth(72)
        self.chance.setToolTip("How likely this effect is for an image (in “a few at random”: how often it is picked "
                               "compared with the others)")
        self.chance.valueChanged.connect(lambda _v: self.changed.emit())
        self.eye = QToolButton()
        self.eye.setIcon(utils.newIcon("eye"))
        self.eye.setCheckable(True)
        self.eye.setAutoRaise(True)
        self.eye.setToolTip("Preview this effect alone, across its whole range")
        self.eye.toggled.connect(lambda on: self.solo.emit(self.key, on))
        g.addWidget(self.on, 0, 0, 1, 3)
        g.addWidget(self.chance, 0, 3)
        g.addWidget(self.eye, 0, 4)
        g.setColumnStretch(0, 1)
        self.param_widgets = []
        for r, p in enumerate(op["params"], 1):
            lab = QLabel(p["label"])
            utils.style(lab, "QLabel{color:@muted;}")
            lo, hi = self._spin(p), self._spin(p)
            dash = QLabel("to")
            utils.style(dash, "QLabel{color:@muted;}")
            g.addWidget(lab, r, 0)
            g.addWidget(lo, r, 1)
            g.addWidget(dash, r, 2, Qt.AlignmentFlag.AlignCenter)
            g.addWidget(hi, r, 3, 1, 2)
            self.spins[p["name"]] = (lo, hi)
            self.param_widgets += [lab, lo, dash, hi]
            lo.valueChanged.connect(lambda _v, a=lo, b=hi: self._order(a, b, True))
            hi.valueChanged.connect(lambda _v, a=lo, b=hi: self._order(a, b, False))
        self._toggled(False)

    def _spin(self, p):
        if p["dec"] == 0:
            s = QSpinBox()
            s.setRange(int(p["lo"]), int(p["hi"]))
        else:
            s = QDoubleSpinBox()
            s.setDecimals(int(p["dec"]))
            s.setRange(float(p["lo"]), float(p["hi"]))
            s.setSingleStep(max(10 ** -p["dec"], (p["hi"] - p["lo"]) / 40.0))
        s.setSuffix(p["unit"])
        s.setMinimumWidth(78)
        s.setKeyboardTracking(False)
        return s

    def _order(self, lo, hi, lo_moved: bool):
        if lo.value() > hi.value():                       # the range stays a range
            (hi if lo_moved else lo).blockSignals(True)
            (hi.setValue(lo.value()) if lo_moved else lo.setValue(hi.value()))
            (hi if lo_moved else lo).blockSignals(False)
        self.changed.emit()

    def _toggled(self, on: bool):
        for w in self.param_widgets:
            w.setVisible(on)
        self.chance.setEnabled(on)
        self.changed.emit()

    def set(self, cfg: dict):
        for w in [self.on, self.chance] + [s for pair in self.spins.values() for s in pair]:
            w.blockSignals(True)
        self.on.setChecked(bool(cfg.get("on")))
        self.chance.setValue(int(round(float(cfg.get("p", 0.3)) * 100)) or 1)
        for name, (lo, hi) in self.spins.items():
            a, b = cfg.get(name, [lo.minimum(), hi.maximum()])
            lo.setValue(type(lo.value())(a))
            hi.setValue(type(hi.value())(b))
        for w in [self.on, self.chance] + [s for pair in self.spins.values() for s in pair]:
            w.blockSignals(False)
        for w in self.param_widgets:
            w.setVisible(self.on.isChecked())
        self.chance.setEnabled(self.on.isChecked())

    def get(self) -> dict:
        d = {"on": self.on.isChecked(), "p": self.chance.value() / 100.0}
        for name, (lo, hi) in self.spins.items():
            d[name] = [lo.value(), hi.value()]
        return d

    def set_available(self, ok: bool, why: str = ""):
        self.on.setEnabled(ok)
        self.eye.setEnabled(ok)
        self.on.setToolTip(self.op["tip"] if ok else f"{self.op['tip']} — {why}")
        if not ok:
            self.on.blockSignals(True)
            self.on.setChecked(False)
            self.on.blockSignals(False)
            self._toggled(False)


class FamilyBox(QWidget):
    changed = pyqtSignal()
    install = pyqtSignal(str)                              # "install this library" (the two library families)

    def __init__(self, key: str, title: str, rows: list):
        super().__init__()
        self.key, self.title, self.rows = key, title, rows
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setObjectName("famHead")
        utils.style(self.toggle, "#famHead{font-weight:600; padding:5px 6px; border:0; text-align:left;}"
                                 "#famHead:hover{background:@pill; border-radius:6px;}")
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.toggled.connect(self._open)
        head.addWidget(self.toggle, 1)
        self.none = QToolButton()
        self.none.setText("None")
        self.none.setAutoRaise(True)
        self.none.setToolTip("Switch this whole family off")
        self.none.clicked.connect(self._all_off)
        head.addWidget(self.none)
        lay.addLayout(head)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        utils.style(self.note, "QLabel{color:@muted; padding:0 8px 4px 8px;}")
        self.note.setVisible(False)
        lay.addWidget(self.note)
        self.install_btn = utils.newButton(f"Install {key}", icon="import", slot=lambda: self.install.emit(key),
                                           tip=f"pip install {key} into the Python that runs AIO QC")
        self.install_btn.setVisible(False)
        irow = QHBoxLayout()
        irow.setContentsMargins(8, 0, 0, 4)
        irow.addWidget(self.install_btn)
        irow.addStretch(1)
        lay.addLayout(irow)
        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(4, 0, 0, 6)
        bl.setSpacing(0)
        for r in rows:
            bl.addWidget(r)
            r.changed.connect(self._row_changed)
        lay.addWidget(self.body)
        self._open(False)
        self.refresh()

    def _open(self, on: bool):
        self.body.setVisible(on)
        self.toggle.setIcon(utils.newIcon("chevron-up" if on else "chevron-down"))

    def _all_off(self):
        for r in self.rows:
            r.on.setChecked(False)

    def _row_changed(self):
        self.refresh()
        self.changed.emit()

    def refresh(self):
        n = sum(1 for r in self.rows if r.on.isChecked())
        title = self.title.replace("&", "&&")              # (a single & would be read as a shortcut marker)
        self.toggle.setText(f"{title}  ·  {n} of {len(self.rows)} on" if n else f"{title}  ·  {len(self.rows)}")
        self.none.setEnabled(n > 0)

    def filter(self, text: str):
        text = text.strip().lower()
        hits = 0
        for r in self.rows:
            ok = not text or text in r.op["title"].lower() or text in r.op["tip"].lower() or text in r.key
            r.setVisible(ok)
            hits += ok
        self.setVisible(hits > 0)
        if text and hits:
            self.toggle.setChecked(True)


class Thumb(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("augThumb")
        utils.style(self, "#augThumb{background:@surface; border:1px solid @line; border-radius:8px;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(3)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic.setMinimumSize(160, 110)
        self.pic.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.cap = QLabel("")
        utils.style(self.cap, "QLabel{color:@muted;}")
        self.cap.setWordWrap(False)
        lay.addWidget(self.pic, 1)
        lay.addWidget(self.cap)
        self._pm = None

    def set(self, pm: QPixmap | None, caption: str, tip: str = ""):
        self._pm = pm
        self.cap.setText(caption)
        self.cap.setToolTip(tip or caption)
        self.pic.setToolTip(tip or caption)
        self._fit()

    def _fit(self):
        if self._pm is None or self._pm.isNull():
            self.pic.clear()
            return
        dpr = self.devicePixelRatioF()
        pm = self._pm.scaled(self.pic.size() * dpr, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self.pic.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


# ====================================================================================================== the window
class AugmentDialog(QDialog):
    TITLE = "Image Augmentation"
    SUPPORTED = None
    progressed = pyqtSignal(int, int, dict)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(utils.newIcon("tool-augment"))
        from AIO_QC.utils.qt_utils import fit_size, scrolled
        fit_size(self, "large")
        self.settings = utils.app_settings()
        self.workers: list = []
        self.cancel = threading.Event()
        self._loading = False
        self._solo: str | None = None
        self._seed = 1
        self._gen = 0
        self._busy = False
        self._again = False
        self._src = None                     # (image BGR, shapes px, caption)
        self._items = None                   # scan of the open dataset (cached)
        self._only_classes: set = set()
        self._stored: dict | None = None     # the recipe as last loaded / saved (to know when it was edited)
        self._lib_cfg: dict = {}             # library effects of the recipe (kept while their library is missing)
        self._lib_ok: set = set()            # library effects that work in this Python
        self._donor_cache: dict = {}         # images objects are pasted from in the preview
        self._src_key = None
        self._job_t0 = 0.0

        # ------------------------------------------------------------------ left: the recipe
        top = QHBoxLayout()
        self.recipes = QComboBox()
        self.recipes.setMinimumContentsLength(14)
        self.recipes.activated.connect(lambda _i: self.load_named(self.recipes.currentData()))
        top.addWidget(self.recipes, 1)
        self.save_btn = utils.newButton("Save", icon="save", slot=self.save, tip="Save this recipe (asks for a name "
                                                                                 "when it is a built-in preset)")
        top.addWidget(self.save_btn)
        more = QToolButton()
        more.setText("⋯")
        more.setToolTip("Save as · Import · Export · Delete")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(more)
        m.addAction("Save as…", lambda: self.save(ask=True))
        m.addAction("Import a recipe or an aug_parameters.yaml…", self.import_recipe)
        m.addAction("Export this recipe…", self.export_recipe)
        m.addSeparator()
        self.a_delete = m.addAction("Delete this saved recipe", self.delete_recipe)
        more.setMenu(m)
        top.addWidget(more)
        self.recipe_note = QLabel("")
        self.recipe_note.setWordWrap(True)
        utils.style(self.recipe_note, "QLabel{color:@muted;}")

        how = QGroupBox("How much")
        hf = QFormLayout(how)
        self.mode = QComboBox()
        self.mode.addItem("A few effects at random (recommended)", "some")
        self.mode.addItem("Every effect by its own chance", "each")
        self.mode.addItem("One effect per image (classic)", "one")
        self.mode.setItemData(0, "Each image gets a few different effects, the likelier ones more often",
                              Qt.ItemDataRole.ToolTipRole)
        self.mode.setItemData(1, "Every ticked effect is tried with its chance: more effects ticked, more stacked",
                              Qt.ItemDataRole.ToolTipRole)
        self.mode.setItemData(2, "One family at random, one of its effects, by its chance — the older pipeline",
                              Qt.ItemDataRole.ToolTipRole)
        self.mode.currentIndexChanged.connect(self._changed)
        self.n_min, self.n_max = QSpinBox(), QSpinBox()
        for s in (self.n_min, self.n_max):
            s.setRange(0, 12)
            s.valueChanged.connect(self._changed)
        self.n_min.setToolTip("0: some images stay unchanged")
        nrow = QHBoxLayout()
        nrow.addWidget(self.n_min)
        nrow.addWidget(QLabel("to"))
        nrow.addWidget(self.n_max)
        nrow.addWidget(QLabel("per image"))
        nrow.addStretch(1)
        self.n_box = QWidget()
        self.n_box.setLayout(nrow)
        nrow.setContentsMargins(0, 0, 0, 0)
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(25, 200)
        self.strength.setSingleStep(5)
        self.strength.setToolTip("Scales every effect towards (below 100 %) or away from (above) “no change”")
        self.strength_lab = QLabel("100 %")
        self.strength.valueChanged.connect(lambda v: (self.strength_lab.setText(f"{v} %"), self._changed()))
        srow = QHBoxLayout()
        srow.addWidget(self.strength, 1)
        srow.addWidget(self.strength_lab)
        hf.addRow("Effects", self.mode)
        hf.addRow("", self.n_box)
        hf.addRow("Strength", srow)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Find an effect …  (blur, rain, jpeg)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda t: [f.filter(t) for f in self.families])

        self.rows: dict[str, OpRow] = {}
        self.families: list[FamilyBox] = []
        fam_host = QWidget()
        fl = QVBoxLayout(fam_host)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(2)
        for fkey, ftitle in G.FAMILIES:
            rows = []
            for o in G.CATALOG:
                if o["family"] != fkey:
                    continue
                r = OpRow(o)
                r.solo.connect(self._solo_toggled)
                self.rows[o["key"]] = r
                rows.append(r)
            fb = FamilyBox(fkey, ftitle, rows)
            fb.changed.connect(self._changed)
            fb.install.connect(self._install_library)
            self.families.append(fb)
            fl.addWidget(fb)
        self.families[0].toggle.setChecked(True)

        lab_box = QGroupBox("Labels after geometry")
        lf = QFormLayout(lab_box)
        self.min_visible = QSpinBox()
        self.min_visible.setRange(0, 100)
        self.min_visible.setSuffix(" % visible")
        self.min_visible.setToolTip("An object pushed partly out of the image keeps its label only when at least this "
                                    "much of it is still inside (its shape is clipped to the image)")
        self.min_size = QSpinBox()
        self.min_size.setRange(0, 200)
        self.min_size.setSuffix(" px")
        self.min_size.setToolTip("Objects that end up smaller than this (either side) lose their label")
        self.border = QComboBox()
        for k, t in G.BORDERS.items():
            self.border.addItem(t, k)
        self.border.setToolTip("What fills the area a rotation / shift uncovers. Grey is what YOLO pads with; "
                               "Mirrored shows unlabelled copies of objects — avoid it for detection")
        for w in (self.min_visible, self.min_size):
            w.valueChanged.connect(self._changed)
        self.border.currentIndexChanged.connect(self._changed)
        lf.addRow("Keep an object with", self.min_visible)
        lf.addRow("Drop when under", self.min_size)
        lf.addRow("Uncovered area", self.border)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.addLayout(top)
        ll.addWidget(self.recipe_note)
        ll.addWidget(how)
        ll.addWidget(self.search)
        ll.addWidget(fam_host)
        ll.addWidget(lab_box)
        ll.addStretch(1)

        # ------------------------------------------------------------------ middle: preview
        bar = QHBoxLayout()
        self.prev_btn = utils.newButton("", icon="prev", slot=lambda: self._step(-1), tip="Previous image")
        self.next_btn = utils.newButton("", icon="next", slot=lambda: self._step(1), tip="Next image")
        self.img_lab = QLabel("")
        self.img_lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.shuffle = utils.newButton("Shuffle", icon="reset-default", slot=self._shuffle,
                                       tip="Other random draws of the same recipe (R)")
        self.shuffle.setShortcut("R")
        self.show_labels = QCheckBox("Labels")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(lambda _on: self._schedule())
        self.speed = QLabel("")
        utils.style(self.speed, "QLabel{color:@muted;}")
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.next_btn)
        bar.addWidget(self.img_lab, 1)
        bar.addWidget(self.speed)
        bar.addWidget(self.show_labels)
        bar.addWidget(self.shuffle)
        self.thumbs = [Thumb() for _ in range(N_VARIATIONS + 1)]
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, t in enumerate(self.thumbs):
            grid.addWidget(t, i // 3, i % 3)
        for r in range(2):
            grid.setRowStretch(r, 1)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        self.solo_note = QLabel("")
        utils.style(self.solo_note, "QLabel{color:@accent; font-weight:600;}")
        self.solo_note.setVisible(False)

        # ------------------------------------------------------------------ bottom: use it
        self.tabs = QTabWidget()
        self.tabs.addTab(self._train_tab(), "Train with it (no files)")
        self.tabs.addTab(self._generate_tab(), "Generate files")
        self.tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.tabs.currentChanged.connect(lambda _i: self._schedule())     # (pasting shows on the Generate tab)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.addLayout(bar)
        rl.addWidget(self.solo_note)
        rl.addLayout(grid, 1)
        rl.addWidget(self.tabs)

        split = QSplitter(Qt.Orientation.Horizontal)
        left_sa = scrolled(left)
        left_sa.setMinimumWidth(left.minimumSizeHint().width() + left_sa.verticalScrollBar().sizeHint().width() + 2)
        split.addWidget(left_sa)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 7)
        split.setChildrenCollapsible(False)
        split.setSizes([470, 1100])
        foot = QHBoxLayout()
        self.foot_note = QLabel("")
        utils.style(self.foot_note, "QLabel{color:@muted;}")
        foot.addWidget(self.foot_note, 1)
        foot.addWidget(utils.newButton("Help", icon="question", slot=lambda: self._help()))
        foot.addWidget(utils.newButton("Close", slot=self.close))
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addLayout(foot)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(160)
        self.timer.timeout.connect(self._render)
        self.est_timer = QTimer(self)
        self.est_timer.setSingleShot(True)
        self.est_timer.setInterval(250)
        self.est_timer.timeout.connect(self._estimate)
        self.progressed.connect(self._on_progress)
        self._seen = (None, None)                          # (dataset, shown file) the window last followed
        self.follow = QTimer(self)
        self.follow.setInterval(700)
        self.follow.timeout.connect(self._follow)

        self._fill_recipes()
        last = self.settings.value("augment/last", "") or ""
        try:
            self.set_recipe(json.loads(last) if last else G.default_recipe(), stored=False)
        except (TypeError, ValueError):
            self.set_recipe(G.default_recipe(), stored=False)
        name = self.recipe()["name"]
        i = self.recipes.findData(name)
        self.recipes.setCurrentIndex(i if i >= 0 else 0)
        self._stored = A.load_recipe(name)
        self._check_libraries()
        self.refresh_dataset()

    # ================================================================== tabs
    def _train_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        t = QLabel("Train Model applies this recipe to every training image <b>in memory</b>, differently each epoch: "
                   "nothing is written, so it takes no disk space however large the dataset is. It runs after "
                   "Ultralytics' own mosaic / flips (set under <i>Augmentation and learning rate</i> there), on the "
                   "training split only; boxes and masks follow the geometry.")
        t.setWordWrap(True)
        t.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(t)
        r = QHBoxLayout()
        self.share = QSpinBox()
        self.share.setRange(5, 100)
        self.share.setSingleStep(5)
        self.share.setValue(int(self.settings.value("augment/share", 100)))
        self.share.setSuffix(" % of the images")
        self.share.setToolTip("The rest of the images go through as they are (each epoch another part)")
        r.addWidget(QLabel("Apply to"))
        r.addWidget(self.share)
        r.addStretch(1)
        self.train_btn = utils.newButton("Use in Train Model…", icon="tool-train", slot=self.use_in_training)
        self.train_btn.setProperty("variant", "primary")
        r.addWidget(self.train_btn)
        v.addLayout(r)
        self.train_note = QLabel("")
        self.train_note.setWordWrap(True)
        utils.style(self.train_note, "QLabel{color:@muted;}")
        v.addWidget(self.train_note)
        return w

    def _generate_tab(self) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        g.setHorizontalSpacing(10)
        self.ds_lab = QLabel("")
        self.ds_lab.setWordWrap(True)
        g.addWidget(self.ds_lab, 0, 0, 1, 4)
        self.copies = QSpinBox()
        self.copies.setRange(1, 50)
        self.copies.setValue(int(self.settings.value("augment/copies", 2)))
        self.copies.setSuffix(" per image")
        self.only_train = QCheckBox("Only the train split")
        self.only_train.setChecked(True)
        self.only_train.setToolTip("val / test must stay as they are — they measure the model on real images")
        self.classes_btn = QToolButton()
        self.classes_btn.setText("All classes")
        self.classes_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.classes_btn.setToolTip("Augment only the images that hold these classes")
        self.balance = QCheckBox("More copies for rare classes, up to")
        self.balance.setToolTip("An image with a rare class gets more copies (most common class count ÷ its class "
                                "count, capped), so the classes come closer")
        self.cap = QSpinBox()
        self.cap.setRange(2, 20)
        self.cap.setValue(5)
        self.cap.setPrefix("× ")
        self.originals = QComboBox()
        self.originals.addItem("linked into the new folder", "link")
        self.originals.addItem("referred to in dataset.yaml (nothing copied)", "reference")
        self.originals.addItem("left out", "none")
        for i, tip in enumerate((
                "The output is a complete dataset on its own. Hard links: no extra space on the same drive (copies "
                "on another drive)",
                "YOLO: the new dataset.yaml lists the original folders next to the new one (absolute paths) — "
                "nothing is copied or linked, on any drive",
                "Only the augmented images (a YOLO dataset.yaml still points val / test at the real images)")):
            self.originals.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.originals.setToolTip("Where the untouched images are when you train on the result")
        # objects cut from other images, pasted in first
        self.paste = QCheckBox("Paste objects from other images")
        self.paste.setToolTip("Before the effects: objects cut out of other images of this dataset (by their "
                              "polygon) are pasted where they do not cover what is labelled already — more "
                              "instances per image, rare classes in new places. Files only (Ultralytics has its "
                              "own copy-paste for segmentation while training)")
        self.paste_p = QSpinBox()
        self.paste_p.setRange(1, 100)
        self.paste_p.setValue(int(self.settings.value("augment/paste_p", 50)))
        self.paste_p.setSuffix(" % of images")
        self.paste_lo, self.paste_hi = QSpinBox(), QSpinBox()
        for sp, v in ((self.paste_lo, 1), (self.paste_hi, 3)):
            sp.setRange(1, 20)
            sp.setValue(v)
        self.paste_s_lo, self.paste_s_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.paste_s_lo, 0.7), (self.paste_s_hi, 1.2)):
            sp.setRange(0.1, 4.0)
            sp.setSingleStep(0.1)
            sp.setDecimals(1)
            sp.setValue(v)
            sp.setSuffix("×")
            sp.setToolTip("Size, compared with the size the object had in its own image")
        self.paste_overlap = QSpinBox()
        self.paste_overlap.setRange(0, 100)
        self.paste_overlap.setSuffix(" % overlap")
        self.paste_overlap.setToolTip("How much a pasted object may cover an existing one (0: never)")
        self.paste_boxes = QCheckBox("boxes too")
        self.paste_boxes.setToolTip("Rectangles are pasted as rectangular cut-outs (with their background). "
                                    "Polygons are always cut exactly")
        self.paste_classes = QToolButton()
        self.paste_classes.setText("any class")
        self.paste_classes.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.paste_classes.setToolTip("Paste only objects of these classes (the rare ones, usually)")
        self._paste_classes: set = set()
        self.paste_row = [self.paste_p, self.paste_lo, self.paste_hi, self.paste_s_lo, self.paste_s_hi,
                          self.paste_overlap, self.paste_boxes, self.paste_classes]
        self.max_side = QSpinBox()
        self.max_side.setRange(0, 16000)
        self.max_side.setSingleStep(32)
        self.max_side.setSpecialValueText("as it is")
        self.max_side.setSuffix(" px")
        self.max_side.setToolTip("Shrink the longer side to this before augmenting (smaller files, faster)")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setToolTip("The same seed and settings give exactly the same files")
        self.n_workers = QSpinBox()
        self.n_workers.setRange(0, 61)
        self.n_workers.setSpecialValueText("auto")
        self.n_workers.setToolTip("Processes working in parallel (auto: one per CPU core, minus one)")
        self.out = QLineEdit()
        self.out_btn = utils.newButton("Browse…", icon="open", slot=self._browse_out)
        self.est = QLabel("")
        self.est.setWordWrap(True)
        utils.style(self.est, "QLabel{color:@muted;}")
        self.go = utils.newButton("Generate", icon="tool-augment", slot=self.generate)
        self.go.setProperty("variant", "primary")
        self.stop = utils.newButton("Stop", slot=self.cancel.set, tip="Stop now — what is written stays, and "
                                                                       "Generate later goes on from there")
        self.stop.setVisible(False)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.TextFormat.RichText)
        self.open_btn = utils.newButton("Open the result in AIO QC", icon="open", slot=self._open_result)
        self.folder_btn = utils.newButton("Open folder", slot=self._open_folder)
        for b in (self.open_btn, self.folder_btn):
            b.setVisible(False)
        g.addWidget(QLabel("Copies"), 1, 0)
        g.addWidget(self.copies, 1, 1)
        g.addWidget(self.only_train, 1, 2)
        g.addWidget(self.classes_btn, 1, 3)
        r2 = QHBoxLayout()
        r2.addWidget(self.balance)
        r2.addWidget(self.cap)
        r2.addSpacing(16)
        r2.addWidget(QLabel("Originals"))
        r2.addWidget(self.originals)
        r2.addStretch(1)
        g.addLayout(r2, 2, 0, 1, 4)
        rp = QHBoxLayout()
        rp.addWidget(self.paste)
        rp.addWidget(self.paste_p)
        rp.addWidget(self.paste_lo)
        self._paste_labs = [QLabel("to"), QLabel("objects, size"), QLabel("to")]
        rp.addWidget(self._paste_labs[0])
        rp.addWidget(self.paste_hi)
        rp.addWidget(self._paste_labs[1])
        rp.addWidget(self.paste_s_lo)
        rp.addWidget(self._paste_labs[2])
        rp.addWidget(self.paste_s_hi)
        rp.addWidget(self.paste_overlap)
        rp.addWidget(self.paste_boxes)
        rp.addWidget(self.paste_classes)
        rp.addStretch(1)
        self.paste_host = QWidget()
        self.paste_host.setLayout(rp)
        rp.setContentsMargins(0, 0, 0, 0)
        g.addWidget(self.paste_host, 3, 0, 1, 4)
        r3 = QHBoxLayout()
        for lab, wdg in (("Longer side", self.max_side), ("Seed", self.seed), ("Workers", self.n_workers)):
            r3.addWidget(QLabel(lab))
            r3.addWidget(wdg)
            r3.addSpacing(10)
        r3.addStretch(1)
        g.addLayout(r3, 4, 0, 1, 4)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Save to"))
        r4.addWidget(self.out, 1)
        r4.addWidget(self.out_btn)
        g.addLayout(r4, 5, 0, 1, 4)
        r5 = QHBoxLayout()
        r5.addWidget(self.est, 1)
        r5.addWidget(self.stop)
        r5.addWidget(self.go)
        g.addLayout(r5, 6, 0, 1, 4)
        g.addWidget(self.bar, 7, 0, 1, 4)
        r7 = QHBoxLayout()
        r7.addWidget(self.result, 1)
        r7.addWidget(self.folder_btn)
        r7.addWidget(self.open_btn)
        g.addLayout(r7, 8, 0, 1, 4)
        g.setColumnStretch(3, 1)
        for wdg in (self.copies, self.cap, self.max_side):
            wdg.valueChanged.connect(lambda _v: self.est_timer.start())
        for wdg in (self.only_train, self.balance):
            wdg.toggled.connect(lambda _on: self.est_timer.start())
        self.originals.currentIndexChanged.connect(lambda _i: self.est_timer.start())
        self.paste.toggled.connect(self._paste_changed)
        self.paste_boxes.toggled.connect(self._paste_changed)
        for sp in (self.paste_p, self.paste_lo, self.paste_hi, self.paste_s_lo, self.paste_s_hi, self.paste_overlap):
            sp.valueChanged.connect(self._paste_changed)
        self._paste_changed()
        self.out.textChanged.connect(lambda _t: self.est_timer.start())
        return w

    def paste_cfg(self) -> dict | None:
        """The paste settings (None: off, or not for this kind of dataset)."""
        if not self.paste.isChecked() or not self.paste.isEnabled():
            return None
        return A.clean_paste({"p": self.paste_p.value() / 100.0,
                              "count": [self.paste_lo.value(), self.paste_hi.value()],
                              "scale": [self.paste_s_lo.value(), self.paste_s_hi.value()],
                              "overlap": self.paste_overlap.value() / 100.0, "boxes": self.paste_boxes.isChecked(),
                              "classes": sorted(self._paste_classes) or None})

    def _paste_changed(self, *_):
        on = self.paste.isChecked() and self.paste.isEnabled()
        for w in self.paste_row + getattr(self, "_paste_labs", []):
            w.setVisible(on)
        if hasattr(self, "timer"):
            self.settings.setValue("augment/paste_p", self.paste_p.value())
            self._schedule()                               # the preview shows the pasting too
            self.est_timer.start()

    def _paste_class_toggled(self, c, on):
        (self._paste_classes.add if on else self._paste_classes.discard)(c)
        n = len(self._paste_classes)
        self.paste_classes.setText("any class" if not n else (next(iter(self._paste_classes)) if n == 1
                                                               else f"{n} classes"))
        self._paste_changed()

    # ================================================================== recipes
    def _fill_recipes(self, select: str | None = None):
        cur = select or self.recipes.currentData()
        self.recipes.blockSignals(True)
        self.recipes.clear()
        saved = A.recipe_names()
        pre = G.presets()
        for n in saved:
            self.recipes.addItem(n, n)
            self.recipes.setItemData(self.recipes.count() - 1, "A recipe you saved", Qt.ItemDataRole.ToolTipRole)
        if saved:
            self.recipes.insertSeparator(self.recipes.count())
        for n, (_r, why) in pre.items():
            if n in saved:
                continue
            self.recipes.addItem(f"{n}  (preset)", n)
            self.recipes.setItemData(self.recipes.count() - 1, why, Qt.ItemDataRole.ToolTipRole)
        i = self.recipes.findData(cur)
        if i >= 0:
            self.recipes.setCurrentIndex(i)
        self.recipes.blockSignals(False)

    def load_named(self, name: str | None):
        r = A.load_recipe(name) if name else None
        if r is None:
            return
        self.set_recipe(r)
        i = self.recipes.findData(name)                    # (also when asked for from Train Model ▸ Edit…)
        if i < 0:
            self._fill_recipes(select=name)
        elif i != self.recipes.currentIndex():
            self.recipes.blockSignals(True)
            self.recipes.setCurrentIndex(i)
            self.recipes.blockSignals(False)

    def set_recipe(self, recipe: dict, stored: bool = True):
        r = G.clean(recipe)
        self._loading = True
        try:
            self._name = r["name"]
            self.mode.setCurrentIndex(max(0, self.mode.findData(r["mode"])))
            self.n_min.setValue(r["min"])
            self.n_max.setValue(r["max"])
            self.strength.setValue(int(round(r["strength"] * 100)))
            self.min_visible.setValue(int(round(r["min_visible"] * 100)))
            self.min_size.setValue(int(round(r["min_size"])))
            self.border.setCurrentIndex(max(0, self.border.findData(r["border"])))
            self._lib_cfg = {k: dict(v) for k, v in r["ops"].items() if G.OPS[k]["needs"]}
            for k, row in self.rows.items():
                if G.OPS[k]["needs"] and k not in self._lib_ok:
                    continue                               # not usable here: the recipe keeps it as it is
                row.set(r["ops"][k])
                if row.eye.isChecked():
                    row.eye.setChecked(False)
            for f in self.families:
                f.refresh()
        finally:
            self._loading = False
        if stored:
            self._stored = self.recipe()
        self._changed()

    def recipe(self) -> dict:
        return G.clean({"name": getattr(self, "_name", "Custom"), "mode": self.mode.currentData(),
                        "min": self.n_min.value(), "max": max(self.n_max.value(), self.n_min.value(), 1),
                        "strength": self.strength.value() / 100.0, "border": self.border.currentData(),
                        "min_visible": self.min_visible.value() / 100.0, "min_size": self.min_size.value(),
                        "ops": {k: (r.get() if not G.OPS[k]["needs"] or k in self._lib_ok
                                    else self._lib_cfg.get(k, r.get())) for k, r in self.rows.items()}})

    def edited(self) -> bool:
        return self._stored is None or self.recipe() != G.clean(self._stored)

    def save(self, ask: bool = False) -> str | None:
        """-> the name it was saved under (None: cancelled)."""
        r = self.recipe()
        name = r["name"]
        if ask or name not in A.recipe_names():
            suggestion = name if name not in G.presets() else f"{name} (mine)"
            name, ok = QInputDialog.getText(self, "Save the recipe", "Name:", text=suggestion)
            name = (name or "").strip()
            if not ok or not name:
                return None
            if name in A.recipe_names() and name != r["name"] and QMessageBox.question(
                    self, self.TITLE, f"Replace the saved recipe “{name}”?") != QMessageBox.StandardButton.Yes:
                return None
        r["name"] = name
        try:
            A.save_recipe(r)
        except OSError as exc:
            QMessageBox.warning(self, self.TITLE, f"Could not save the recipe:\n{exc}")
            return None
        self._name = name
        self._stored = self.recipe()
        self._fill_recipes(select=name)
        self._changed()
        self.foot_note.setText(f"Saved “{name}”.")
        td = getattr(self.app, "train_dialog", None)
        if td is not None and hasattr(td, "hyp"):
            td.hyp.refresh_recipes()
        return name

    def delete_recipe(self):
        name = self.recipes.currentData()
        if name not in A.recipe_names():
            return
        if QMessageBox.question(self, self.TITLE, f"Delete the saved recipe “{name}”?") != \
                QMessageBox.StandardButton.Yes:
            return
        A.delete_recipe(name)
        self._fill_recipes()
        self.load_named(self.recipes.currentData())

    def import_recipe(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import a recipe", "", "Recipes (*.json *.yaml *.yml);;All files (*)")
        if not p:
            return
        try:
            r = A.import_file(p)
        except Exception as exc:
            _ignored(exc)
            QMessageBox.warning(self, self.TITLE, f"Could not read it as a recipe:\n{exc}")
            return
        self.set_recipe(r, stored=False)
        on = sum(1 for v in r["ops"].values() if v["on"])
        self.foot_note.setText(f"Imported “{r['name']}” ({on} effects on) — Save keeps it.")

    def export_recipe(self):
        r = self.recipe()
        p, _ = QFileDialog.getSaveFileName(self, "Export the recipe", f"{A._slug(r['name'])}.json", "Recipe (*.json)")
        if p:
            try:
                Path(p).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, self.TITLE, f"Could not write it:\n{exc}")

    def _changed(self):
        if self._loading:
            return
        self.n_box.setVisible(self.mode.currentData() == "some")
        r = self.recipe()
        on = [k for k, v in r["ops"].items() if v["on"]]
        geo = [k for k in on if G.OPS[k]["kind"] in ("matrix", "dense")]
        saved = self._name in A.recipe_names()
        state = "edited — Save keeps it" if self.edited() else ("saved" if saved else "a built-in preset")
        self.recipe_note.setText(f"“{self._name}” · {state} · {len(on)} effects on"
                                 + (f" ({len(geo)} move the labels)" if geo else ""))
        self.a_delete.setEnabled(saved)
        try:
            self.settings.setValue("augment/last", json.dumps(r))
        except Exception as _exc:
            _ignored(_exc)
        libs = sorted({G.OPS[k]["needs"] for k in on if G.OPS[k]["needs"]})
        self.train_note.setText(("Uses " + " and ".join(libs) + ": the training environment needs them too (Train "
                                 "Model says when it doesn't; those effects are then left out).") if libs else "")
        self.train_btn.setEnabled(bool(on))
        self._schedule()
        self.est_timer.start()

    # ================================================================== libraries
    def _check_libraries(self):
        self._loading = True
        for f in self.families:
            if f.key in ("albumentations", "imgaug"):
                f.note.setText("Checking …")
                f.note.setVisible(True)
                for r in f.rows:
                    r.set_available(False, "checking")
        self._loading = False
        self._run(lambda: G.libraries(), self._libraries_checked, lambda _m: None)

    def _libraries_checked(self, info: dict):
        for f in self.families:
            lib = info.get(f.key)
            if lib is None:
                continue
            if lib["version"]:
                f.note.setText(f"{f.key} {lib['version']} found" + (f" · {len(lib['broken'])} effects do not work "
                                                                    "with this NumPy / OpenCV" if lib["broken"] else ""))
            else:
                f.note.setText(f"Optional: these come from {f.key}, which is not in the Python that runs AIO QC "
                               f"(pip install {f.key}). The built-in effects cover the same ground without it.")
            import sys
            f.install_btn.setVisible(not lib["version"] and not getattr(sys, "frozen", False))
            f.install_btn.setEnabled(True)
            f.install_btn.setText(f"Install {f.key}")
            self._loading = True
            try:
                for r in f.rows:
                    ok = r.key in lib["effects"]
                    r.set_available(ok, lib["broken"].get(r.key) or f"{f.key} is not installed")
                    if ok:                                 # now usable: what the recipe has for it
                        self._lib_ok.add(r.key)
                        if r.key in self._lib_cfg:
                            r.set(self._lib_cfg[r.key])
                f.refresh()
            finally:
                self._loading = False
        self._changed()

    def _install_library(self, name: str):
        """pip install albumentations / imgaug into this Python (in the background), then look again."""
        import sys
        if name not in ("albumentations", "imgaug") or getattr(sys, "frozen", False):
            return
        fam = next(f for f in self.families if f.key == name)
        fam.install_btn.setEnabled(False)
        fam.install_btn.setText("Installing …")
        fam.note.setText(f"Installing {name} with pip — this can take a minute; keep working meanwhile.")

        def work():
            from AIO_QC.core import procs
            r = procs.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", name],
                          timeout=1800, text=True)
            out = ((getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")).strip()
            return getattr(r, "returncode", 1), out

        def done(res):
            code, out = res
            import importlib
            importlib.invalidate_caches()
            G._LIBS.pop(name, None)                        # look for it again
            for o in G.CATALOG:
                if o["needs"] == name:
                    G._BROKEN.pop(o["key"], None)
                    G._INSTANCES.pop(o["key"], None)
            if code != 0:
                last = out.splitlines()[-1] if out else "pip failed"
                QMessageBox.warning(self, self.TITLE, f"{name} could not be installed:\n\n{last[-400:]}\n\n"
                                    f"In a terminal:  {Path(sys.executable).name} -m pip install {name}")
            self._check_libraries()

        self._run(work, done, lambda msg: (QMessageBox.warning(self, self.TITLE, f"{name} could not be installed: "
                                                               f"{msg[-300:]}"), self._check_libraries()))

    # ================================================================== preview
    def refresh_dataset(self):
        """The open dataset changed (or the window opened): the image to preview on, and the Generate tab."""
        s = getattr(self.app, "session", None)
        self._items = None
        fmt = _fmt_of(s)
        has = s is not None and fmt is not None
        for wdg in (self.copies, self.only_train, self.classes_btn, self.balance, self.cap, self.originals,
                    self.max_side, self.seed, self.n_workers, self.out, self.out_btn, self.go):
            wdg.setEnabled(has)
        self.paste.setEnabled(has and fmt in ("yolo", "labelme"))
        self.paste_host.setVisible(fmt != "images")
        i_ref = self.originals.findData("reference")       # only a YOLO dataset has a dataset.yaml to refer from
        self.originals.model().item(i_ref).setEnabled(fmt == "yolo")
        if fmt != "yolo" and self.originals.currentData() == "reference":
            self.originals.setCurrentIndex(0)
        self._paste_classes = set()
        self.paste_classes.setText("any class")
        if has:
            n = len(s.samples)
            self.ds_lab.setText(f"<b>{s.root.name}</b> · {n:,} images · {s.type.title}")
            self.out.setText(str(s.root.parent / f"{s.root.name}_aug"))
            splits = list(s.splits)
            self.only_train.setVisible("train" in splits and len(splits) > 1)
            m = QMenu(self.classes_btn)
            self._only_classes = set()
            for c in sorted(s.label_totals()):
                a = m.addAction(str(c))
                a.setCheckable(True)
                a.toggled.connect(lambda on, c=c: self._class_toggled(c, on))
            self.classes_btn.setMenu(m)
            self.classes_btn.setText("All classes")
            pm = QMenu(self.paste_classes)
            for c in sorted(s.label_totals()):
                a = pm.addAction(str(c))
                a.setCheckable(True)
                a.toggled.connect(lambda on, c=c: self._paste_class_toggled(c, on))
            self.paste_classes.setMenu(pm)
            self.classes_btn.setVisible(bool(s.label_totals()))     # class folders: the folders are the classes
            self.balance.setVisible(True)
            self.cap.setVisible(True)
        else:
            self.ds_lab.setText("Open a detection / segmentation dataset (LabelMe or YOLO) or class folders in Image "
                                "QC to write an augmented copy of it." if s is None else
                                f"{s.type.title} datasets cannot be written augmented yet — the recipe still works "
                                "in Train Model.")
        self._seen = (id(s) if s is not None else None, getattr(self.app, "current_key", None))
        self._load_source()
        self.est_timer.start()

    def _follow(self):
        """The window follows Image QC: another dataset → the Generate tab; another file → the preview."""
        s = getattr(self.app, "session", None)
        now = (id(s) if s is not None else None, getattr(self.app, "current_key", None))
        if now == self._seen or self.bar.isVisible():
            return
        new_ds = now[0] != self._seen[0]
        self._seen = now
        if new_ds:
            self.refresh_dataset()
        else:
            self._load_source()

    def showEvent(self, e):
        super().showEvent(e)
        self.follow.start()

    def _class_toggled(self, c, on):
        (self._only_classes.add if on else self._only_classes.discard)(c)
        n = len(self._only_classes)
        self.classes_btn.setText("All classes" if not n else (next(iter(self._only_classes)) if n == 1 else
                                                              f"{n} classes"))
        self.est_timer.start()

    def _step(self, d: int):
        s = getattr(self.app, "session", None)
        if s is None or not s.samples:
            return
        keys = [x.key for x in s.samples] if len(s.samples) < 200000 else None
        cur = getattr(self, "_src_key", None)
        if keys is None:
            i = (getattr(self, "_src_i", 0) + d) % len(s.samples)
        else:
            i = (keys.index(cur) + d) % len(keys) if cur in keys else 0
        self._src_i = i
        self._load_source(s.samples[i].key)

    def _load_source(self, key: str | None = None):
        import cv2
        s = getattr(self.app, "session", None)
        key = key or getattr(self.app, "current_key", None)
        img, shapes, cap = None, [], ""
        if s is not None and key in getattr(s, "by_key", {}):
            try:
                sample = s.by_key[key]
                ann = s.load(key)
                path = getattr(ann, "image_path", None) or sample.image_path
                if path is not None and Path(path).is_file():
                    img = A._imread(Path(path))
                if img is not None:
                    h, w = img.shape[:2]
                    from AIO_QC.core import inline_edit as IE
                    for sh in ann.shapes:
                        if sh.get("points"):
                            shapes.append({"label": sh.get("label"), "shape_type": sh.get("shape_type") or "polygon",
                                           "points": IE.to_px(ann, sh["points"], w, h)})
                    cap = sample.display
                    self._src_key = key
            except Exception as _exc:
                _ignored(_exc)
                img = None
        if img is None:
            img, shapes = test_card()
            cap = "A test picture (open a dataset to see your own images)"
            self._src_key = None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = np.ascontiguousarray(img[..., :3])
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = img.shape[:2]
        f = min(1.0, PREVIEW_SIDE / float(max(w, h)))
        if f < 1.0:
            img = cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
            shapes = [dict(sh, points=[[x * f, y * f] for x, y in sh["points"]]) for sh in shapes]
        self._src = (img, shapes, cap)
        self.img_lab.setText(cap)
        has = s is not None and len(getattr(s, "samples", [])) > 1
        self.prev_btn.setEnabled(has)
        self.next_btn.setEnabled(has)
        self._schedule()

    def _solo_toggled(self, key: str, on: bool):
        if on:
            for k, r in self.rows.items():
                if k != key and r.eye.isChecked():
                    r.eye.blockSignals(True)
                    r.eye.setChecked(False)
                    r.eye.blockSignals(False)
            self._solo = key
        elif self._solo == key:
            self._solo = None
        self.solo_note.setText(f"Only “{G.OPS[self._solo]['title']}”, from the low to the high end of its range — "
                               "click its eye again for the whole recipe" if self._solo else "")
        self.solo_note.setVisible(bool(self._solo))
        self._schedule()

    def _shuffle(self):
        self._seed += 97
        self._schedule()

    def _schedule(self):
        if self._src is not None:
            self.timer.start()

    def _colors(self) -> dict:
        out = {}
        for name, c in (getattr(self.app, "label_colors", {}) or {}).items():
            try:
                out[str(name)] = (c.blue(), c.green(), c.red())
            except AttributeError:
                pass
        return out

    def _render(self):
        if self._busy:
            self._again = True
            return
        if self._src is None:
            return
        self._busy = True
        self._gen += 1
        gen, img, shapes = self._gen, self._src[0], self._src[1]
        recipe, solo, seed = self.recipe(), self._solo, self._seed
        labels, colors = self.show_labels.isChecked(), self._colors()
        paste, donor, n_donors = self._paste_preview()

        def work():
            t0 = time.perf_counter()
            res = []
            if solo:
                cfg = dict(recipe["ops"][solo], on=True, p=1.0)
                ps = G.OPS[solo]["params"]
                for i in range(N_VARIATIONS):                # low end … high end of every range
                    t = i / max(1, N_VARIATIONS - 1)
                    c = dict(cfg)
                    for p in ps:
                        lo, hi = cfg[p["name"]]
                        v = lo + (hi - lo) * t
                        c[p["name"]] = [v, v]
                    r = G._with({solo: c}, mode="each", strength=recipe["strength"], border=recipe["border"],
                                min_visible=recipe["min_visible"], min_size=recipe["min_size"])
                    out, sh, log = G.Augmenter(r).apply(img, shapes, seed=seed + i)
                    res.append((out, sh, log))
            else:
                aug = G.Augmenter(recipe)
                for i in range(N_VARIATIONS):
                    im2, sh2, n_p = img, shapes, 0
                    if paste:
                        im2, sh2, n_p = A.paste_objects(img, shapes, paste, np.random.default_rng(seed * 977 + i),
                                                        donor, n_donors)
                    out, sh, log = aug.apply(im2, sh2, seed=seed * 1000 + i)
                    res.append((out, sh, ([("Pasted", f"{n_p} object{'s' if n_p != 1 else ''}")] if n_p else [])
                                + list(log)))
            ms = (time.perf_counter() - t0) * 1000 / N_VARIATIONS
            tiles = [(draw_shapes(img, shapes, colors) if labels else img, "Original",
                      f"{len(shapes)} labelled objects")]
            for out, sh, log in res:
                what = G.describe(log)
                kept = f"{len(sh)} of {len(shapes)} objects kept" if shapes else ""
                tiles.append((draw_shapes(out, sh, colors) if labels else out, what,
                              what + (f"\n{kept}" if kept else "")))
            return gen, tiles, ms

        self._run(work, self._rendered, self._render_failed)

    def _paste_preview(self):
        """(paste settings, donor(i), how many) for the preview — the same pasting Generate does, from the open
        dataset's own images (read at preview size; a few are kept)."""
        cfg = self.paste_cfg() if self.tabs.currentIndex() == 1 else None
        s = getattr(self.app, "session", None)
        fmt = _fmt_of(s)
        if cfg is None or s is None or fmt not in ("yolo", "labelme") or self._src_key is None:
            return None, None, 0
        if self._items is None:
            self._items = A.scan(s.samples, s.root, fmt)
        pool = [it for it in self._items[0] if it.ann]
        if not pool:
            return None, None, 0
        if cfg["classes"] and fmt == "yolo":
            names = [str(n) for n in (getattr(getattr(s.base_adapter, "class_map", None), "names", None) or [])]
            cfg["classes"] = [str(names.index(c)) for c in cfg["classes"] if c in names] + cfg["classes"]
        root, cache = Path(s.root), self._donor_cache
        if cache.get("root") != str(root):
            cache.clear()
            cache["root"] = str(root)
            cache["obb"] = A.is_obb(root, pool) if fmt == "yolo" else False
        names = {str(i): str(n) for i, n in enumerate(
            getattr(getattr(s.base_adapter, "class_map", None), "names", None) or [])} if fmt == "yolo" else {}

        def donor(i):
            if i not in cache:
                if len(cache) > 40:
                    for k in [k for k in cache if isinstance(k, int)][:20]:
                        cache.pop(k, None)
                try:
                    im, sh, _d, _n, _o = A.load_item(root, fmt, pool[i].ann, pool[i].img, cache["obb"], PREVIEW_SIDE)
                    if im is not None and fmt == "yolo":               # the preview draws by class name
                        sh = [dict(x, label=names.get(x.get("cls"), x.get("cls"))) for x in sh]
                    cache[i] = (im, sh) if im is not None else None
                except Exception as _exc:
                    _ignored(_exc)
                    cache[i] = None
            return cache[i]
        return cfg, donor, len(pool)

    def _rendered(self, res):
        self._busy = False
        gen, tiles, ms = res
        if gen == self._gen:
            for t, (bgr, cap, tip) in zip(self.thumbs, tiles):
                t.set(_pixmap(bgr), cap, tip)
            h, w = self._src[0].shape[:2]
            self.speed.setText(f"{ms:.0f} ms per image at {w}×{h}")
        if self._again:
            self._again = False
            self._schedule()

    def _render_failed(self, msg):
        self._busy = False
        self.speed.setText(f"⚠ {msg[-120:]}")
        if self._again:
            self._again = False
            self._schedule()

    # ================================================================== training
    def use_in_training(self):
        name = self._name
        if self.edited() or (name not in A.recipe_names() and name not in G.presets()):
            name = self.save()
            if not name:
                return
        self.settings.setValue("augment/share", self.share.value())
        self.app.open_training()
        td = getattr(self.app, "train_dialog", None)
        if td is not None and hasattr(td, "hyp"):
            td.hyp.set_recipe(name, self.share.value())
            td.hyp_drawer.set_open(True)

    # ================================================================== generating
    def _browse_out(self):
        p = QFileDialog.getExistingDirectory(self, "Save the augmented dataset to", self.out.text() or "")
        if p:
            self.out.setText(p)

    def _plan(self):
        """-> (items, copies) for the open dataset and the settings."""
        s = self.app.session
        fmt = _fmt_of(s)
        if self._items is None:
            self._items = A.scan(s.samples, s.root, fmt)
        items, _outside = self._items
        only = None
        by_class = None
        if self._only_classes or self.balance.isChecked():         # one pass over the index, any number of classes
            by_class = {}
            for k, c in s.index.items():
                for lab in c:
                    by_class.setdefault(lab, []).append(k)
        if self._only_classes:
            only = set()
            for c in self._only_classes:
                only.update(by_class.get(c, ()))
        class_keys = by_class if self.balance.isChecked() else None
        splits = {"train"} if (self.only_train.isVisible() and self.only_train.isChecked()) else None
        return items, A.copies_per_item(items, self.copies.value(), only, splits, class_keys,
                                        self.balance.isChecked(), self.cap.value())

    def _out_problem(self) -> str | None:
        s = getattr(self.app, "session", None)
        t = self.out.text().strip()
        if s is None or not t:
            return "Choose where to save"
        out, root = Path(t).expanduser().resolve(), Path(s.root).resolve()
        if out == root or root in out.parents:
            return "Save outside the dataset folder (the copies would be picked up as its own images)"
        if out in root.parents:
            return "That folder contains the dataset — choose a new one"
        return None

    def _estimate(self):
        s = getattr(self.app, "session", None)
        if s is None or _fmt_of(s) is None or self.bar.isVisible():
            self.est.setText("")
            return
        prob = self._out_problem()
        try:
            items, copies = self._plan()
        except Exception as exc:
            _ignored(exc)
            self.est.setText(f"⚠ {exc}")
            return
        n_new, n_img = sum(copies), sum(1 for c in copies if c)
        link = "hardlink"
        mode = self.originals.currentData()
        need = A.estimate_bytes(s.root, items, copies, mode == "link", link)
        txt = f"{n_new:,} new images from {n_img:,} of {len(items):,}"
        txt += {"link": f" + the {len(items):,} originals (linked)", "reference": " · originals referred to, not copied",
                "none": ""}[mode]
        txt += f" · about {need / 2**30:.1f} GB" if need >= 2**30 else f" · about {max(1, need // 2**20):,} MB"
        try:
            import shutil
            p = Path(self.out.text().strip() or s.root)
            while not p.exists() and p != p.parent:
                p = p.parent
            free = shutil.disk_usage(p).free
            txt += f" ({free / 2**30:.0f} GB free)"
            if need > free * 0.95:
                prob = prob or "Not enough free space there"
        except OSError:
            pass
        if not n_new:
            prob = prob or "Nothing to augment with these choices"
        if not any(v["on"] for v in self.recipe()["ops"].values()):
            prob = prob or "Tick at least one effect"
        self.est.setText(txt + (f"<br><span style='color:{utils.hexc('error')}'>{prob}</span>" if prob else ""))
        self.est.setTextFormat(Qt.TextFormat.RichText)
        self.go.setEnabled(not prob)

    def generate(self):
        s = self.app.session
        fmt = _fmt_of(s)
        if s is None or fmt is None or self._out_problem():
            return
        items, copies = self._plan()
        out = Path(self.out.text().strip()).expanduser()
        if out.exists() and any(out.iterdir()) and QMessageBox.question(
                self, self.TITLE, f"{out}\nis not empty.\n\nGo on there? Augmented files that already exist are kept "
                "(a stopped run continues); nothing is deleted.") != QMessageBox.StandardButton.Yes:
            return
        kpt = getattr(getattr(s.base_adapter, "class_map", None), "kpt_shape", None)
        r = self.recipe()
        if kpt and any(v["on"] and G.OPS[k]["kind"] in ("matrix", "dense") for k, v in r["ops"].items()):
            QMessageBox.information(self, self.TITLE, "This is a pose dataset: keypoints cannot follow geometric "
                                    "effects yet. Switch the Geometry family off (None) to generate.")
            return
        self.settings.setValue("augment/copies", self.copies.value())
        mode = self.originals.currentData()
        paste = self.paste_cfg()
        if paste and paste["classes"] and fmt == "yolo":   # YOLO rows carry class ids: the chosen names as ids
            names = [str(n) for n in (getattr(getattr(s.base_adapter, "class_map", None), "names", None) or [])]
            ids = [str(names.index(c)) for c in paste["classes"] if c in names]
            paste["classes"] = (ids + paste["classes"]) or None
        job = A.Job(str(s.root), str(out), fmt, r, seed=self.seed.value(), originals=mode == "link",
                    reference=mode == "reference", paste=paste,
                    obb=A.is_obb(s.root, items) if fmt == "yolo" else False, max_side=self.max_side.value(),
                    workers=self.n_workers.value())
        self.cancel.clear()
        self._job_out = out
        self._job_t0 = time.time()
        total = sum(1 for c in copies if c)
        self.bar.setRange(0, max(1, total))
        self.bar.setValue(0)
        self.bar.setFormat("starting …")
        for wdg, vis in ((self.bar, True), (self.stop, True), (self.go, False), (self.open_btn, False),
                         (self.folder_btn, False)):
            wdg.setVisible(vis)
        self.result.setText("")
        self.est.setText("")
        last = [0.0]
        from AIO_QC.widgets import tasks
        if getattr(self, "_task", None) is not None:
            self._task.finish()
        self._task = tasks.start(f"Image Augmentation · {Path(out).name}", cancel=self.cancel.set)

        def progress(done, tot, rep):
            now = time.time()
            if now - last[0] > 0.15 or done >= tot:       # a few updates a second are plenty
                last[0] = now
                self.progressed.emit(done, tot, dict(written=rep["written"], skipped=rep["skipped"],
                                                     errors=rep.get("n_errors", 0)))

        self._run(lambda: A.run(job, items, copies, progress, self.cancel), self._generated, self._generate_failed)

    def _on_progress(self, done, total, rep):
        self.bar.setRange(0, max(1, total))
        self.bar.setValue(done)
        el = max(0.001, time.time() - self._job_t0)
        rate = done / el
        left = (total - done) / rate if rate > 0 else 0
        m, sec = divmod(int(left), 60)
        hh, m = divmod(m, 60)
        eta = f"{hh} h {m:02d} min" if hh else (f"{m} min {sec:02d} s" if m else f"{sec} s")
        if getattr(self, "_task", None) is not None:
            self._task.update(done / total if total else None, f"{done:,} / {total:,} images · {eta} left")
        self.bar.setFormat(f"{done:,} / {total:,} images · {rep['written']:,} written"
                           + (f" · {rep['skipped']:,} already there" if rep["skipped"] else "")
                           + (f" · {rep['errors']:,} errors" if rep["errors"] else "")
                           + f" · {rate:.0f} images/s · {eta} left")

    def _end_task(self):
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None

    def _generated(self, rep: dict):
        self._end_task()
        for wdg, vis in ((self.bar, False), (self.stop, False), (self.go, True), (self.folder_btn, True),
                         (self.open_btn, True)):
            wdg.setVisible(vis)
        n_err = rep.get("n_errors", 0)
        dropped = rep["shapes_in"] - rep["shapes_out"]
        txt = (f"<b>{'Stopped' if rep['cancelled'] else 'Done'}</b> in {rep['seconds']:.0f} s — "
               f"{rep['written']:,} augmented images written"
               + (f", {rep['skipped']:,} were already there" if rep["skipped"] else "")
               + (f", {rep['originals']:,} original files added" if rep["originals"] else "")
               + (f" · {rep['pasted']:,} objects pasted" if rep.get("pasted") else "")
               + (f" · {dropped:,} of {rep['shapes_in']:,} labels left the image" if dropped else "")
               + (f" · {rep['odd_rows']:,} label rows could not be read" if rep.get("odd_rows") else ""))
        if n_err:
            txt += (f"<br><span style='color:{utils.hexc('error')}'>{n_err:,} images failed</span> — e.g. "
                    + "; ".join(rep["errors"][:2]))
        if rep["cancelled"]:
            txt += "<br>Generate goes on from here (what is written is skipped)."
        self.result.setText(txt)
        self._estimate()

    def _generate_failed(self, msg):
        self._end_task()
        for wdg, vis in ((self.bar, False), (self.stop, False), (self.go, True)):
            wdg.setVisible(vis)
        from .errors import explain_html
        self.result.setText(f"<span style='color:{utils.hexc('error')}'>Could not generate.</span> "
                            + explain_html(msg))

    def _open_result(self):
        out = getattr(self, "_job_out", None)
        if out is not None and out.is_dir():
            self.app.open_dataset(out)

    def _open_folder(self):
        out = getattr(self, "_job_out", None)
        if out is not None and out.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    # ================================================================== plumbing
    def _help(self):
        try:
            self.app.show_guide()
        except Exception as _exc:
            _ignored(_exc)

    def _run(self, fn, done, failed):
        w = FuncWorker(fn, self)
        w.done.connect(done)
        w.failed.connect(failed)
        w.finished.connect(lambda: self.workers.remove(w) if w in self.workers else None)
        w.finished.connect(w.deleteLater)
        self.workers.append(w)
        w.start()

    def sizeHint(self):
        return QSize(1600, 980)

    def closeEvent(self, e):
        self.cancel.set()
        self._end_task()
        for w in list(self.workers):
            try:
                if w.isRunning():
                    w.wait(5000)
            except RuntimeError:
                pass
        self.timer.stop()
        self.est_timer.stop()
        self.follow.stop()
        if hasattr(self.app, "forget_dialog"):
            try:
                self.app.forget_dialog(self)
            except Exception as _exc:
                _ignored(_exc)
        super().closeEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and isinstance(QApplication.focusWidget(),
                                                                          (QSpinBox, QDoubleSpinBox, QLineEdit)):
            return                                         # Enter in a field applies the value, nothing else
        super().keyPressEvent(e)
