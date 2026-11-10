# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Shapes / Review tab tools (dialogs). The tab a tool appears in is defined in app.QC_TOOLS.

    📐 Dimension / Area filter · 🚨 Outliers · 🧠 Smart Filter
"""
from __future__ import annotations

from AIO_QC.core.dataset import DETECTION_TYPES

import math
from collections import Counter, defaultdict

import numpy as np
from loguru import logger

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QRadioButton, QTabWidget, QVBoxLayout, QWidget,
)

from AIO_QC.core.checks import iqr_bounds, shape_measure
from AIO_QC.core.geometry import shape_metrics, to_polygon
from AIO_QC.core.shape_ops import fingerprint
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import hexc, style as apply_style, theme_color
from AIO_QC.utils import later
from .base_scan import BaseScanDialog, ResultRow, dspin, ispin
from .charts import ScatterBrush, StripChart
from .card_gallery import Card, CardView, Panel, Shape, box_points

NO_LIMIT = "∞"


def _spin(maximum=1e9, decimals=2, suffix=""):
    return dspin(0, maximum, 0, decimals, suffix=suffix)


def _labels_combo(session, with_all=True) -> QComboBox:
    box = QComboBox()
    box.setEditable(True)
    if with_all:
        box.addItem("(all labels)", None)
    for label, n in session.label_totals().most_common():
        box.addItem(f"{label}", label)
    return box


# =============================================================================
# 📐 Dimension / area
# =============================================================================
class DimAreaDialog(BaseScanDialog):
    """
    One scan measures EVERY shape; then the scatter (width × height, log–log) is the filter: drag a rectangle
    over the cluster you care about, or type min / max. The table and the crop grid update instantly — no rescan.

    Region: draw a rectangle on an image of one exact size (e.g. 1920 × 1080) to keep only the shapes in that part
    of the picture — only images of exactly that size take part (a fixed camera's pixel = the same place). Combines
    with the size / area / label limits; “Selection…” acts on it like on any other selection.
    """
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Dim / Area", "📐"
    HELP = ("Scan once, then drag a rectangle on the scatter (or type limits) to find shapes of a given size — "
            f"e.g. the cloud of tiny boxes in the bottom-left corner. Max {NO_LIMIT} = no limit. Region: draw where in "
            "the picture (images of one exact size only). Deleting shapes updates everything in place.")
    LIST_FIELDS = ("key", "idx", "label", "box", "norm", "fp")
    ARRAY_FIELDS = {"w": np.float64, "h": np.float64, "a": np.float64, "px1": np.float32, "py1": np.float32,
                    "px2": np.float32, "py2": np.float32, "iw": np.int32, "ih": np.int32, "fid": np.int64}
    COLUMNS = ["Label", "Shape", "Width", "Height", "Area"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = False
    MAX_ROWS = 20000          # table / crop grid show at most this many; the counter always shows the full number

    def build_params(self, form):
        self.label = _labels_combo(self.session)
        self.label.currentIndexChanged.connect(lambda _i: self._apply_timer.start())
        self.unit = QComboBox()
        self.unit.addItems(["pixels", "% of image"])
        self.unit.setToolTip("Changing the unit needs a new scan")
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        self.ranges = {}
        for r, name in enumerate(("Width", "Height", "Area")):
            lo, hi = _spin(), _spin()
            hi.setSpecialValueText(NO_LIMIT)
            for sp in (lo, hi):
                sp.valueChanged.connect(lambda _v: self._on_spins())
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(QLabel("min"), r, 1)
            grid.addWidget(lo, r, 2)
            grid.addWidget(QLabel("max"), r, 3)
            grid.addWidget(hi, r, 4)
            self.ranges[name] = (lo, hi)
        form.addRow("Label:", self.label)
        form.addRow("Unit:", self.unit)
        form.addRow(grid_w)
        self.add_action("↺ Clear limits", self._clear_limits, "Show every shape")
        self.sel_btn = self.add_action("🧹 Selection…", self._act_on_selection,
                                       "Delete, relabel or flag EVERY shape inside the drawn rectangle (and the "
                                       "Area / Label limits) — not just the rows shown in the table")
        self.add_action("🗑 Delete ticked", self._delete_ticked, "Delete only the ticked (or selected) shapes")
        self.undo_btn = self.add_action("↶ Undo", self._undo, "Put back the files changed by the last delete / relabel")
        self.undo_btn.setEnabled(False)
        self._last_backup = None
        from PyQt6.QtCore import QTimer
        self._apply_timer = QTimer(self, singleShot=True, interval=120, timeout=self._apply_ranges)
        self.data = None

    def post_init(self):
        from PyQt6.QtWidgets import QSplitter
        self.init_marks()
        self.scatter = ScatterBrush()
        self.scatter.selectionChanged.connect(self._on_brush)
        self.cards = CardView(self, n_panels=1)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Drag a rectangle to select shapes by size · click the plot to clear")
        apply_style(hint, "color:@muted;")
        hint.setWordWrap(True)                     # never forces the pane wider
        ll.addWidget(hint)
        ll.addWidget(self.scatter, 1)
        self.views = QTabWidget()
        self.views.addTab(left, "Size")
        self.views.addTab(self._build_region_panel(), "Region")
        self.views.setTabToolTip(1, "Keep only the shapes in a part of the picture — images of one exact size")
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.views)
        split.addWidget(self.cards)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        later(0, split, lambda: split.setSizes([max(1, split.width() // 2)] * 2))     # half / half
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(split, "Visual")
        self.tabs.addTab(self.table, "Table")
        self.table_host.addWidget(self.tabs)
        self.filter_box.currentIndexChanged.connect(lambda _i: self._refresh_cards())

    # ------------------------------------------------------------------ region view (images of one exact size)
    def _build_region_panel(self) -> QWidget:
        from PyQt6.QtCore import QTimer
        from AIO_QC.core.region import MODES
        from AIO_QC.utils import newButton
        from .region_canvas import RegionCanvas
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        from PyQt6.QtWidgets import QSizePolicy
        shrink = QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        row1 = QHBoxLayout()
        self.region_size_box = QComboBox()
        self.region_size_box.setSizeAdjustPolicy(shrink)          # long texts must not widen the pane
        self.region_size_box.setMinimumContentsLength(12)
        self.region_size_box.setToolTip("Only images of exactly this size take part in the region filter")
        self.region_size_box.currentIndexChanged.connect(lambda _i: self._on_region_size())
        self.region_mode = QComboBox()
        self.region_mode.setSizeAdjustPolicy(shrink)
        self.region_mode.setMinimumContentsLength(10)
        for k, text in MODES.items():
            self.region_mode.addItem(text, k)
        self.region_mode.setToolTip("How a shape must relate to the region (its bounding box is used)")
        self.region_mode.currentIndexChanged.connect(lambda _i: self._region_commit())
        self.region_invert = QCheckBox("Outside instead")
        self.region_invert.setToolTip("Select the shapes of these images that are NOT in the region — e.g. everything "
                                      "outside a region of interest")
        self.region_invert.toggled.connect(lambda _on: self._region_commit())
        row1.addWidget(QLabel("Size:"))
        row1.addWidget(self.region_size_box, 3)
        row1.addWidget(self.region_mode, 2)
        row1.addWidget(self.region_invert)
        row2 = QHBoxLayout()
        self.region_spins = []
        for name in ("x", "y", "w", "h"):
            sp = ispin(0, 100000, 0)
            sp.setSuffix(" px")
            sp.setToolTip(f"Region {name} in image pixels")
            sp.editingFinished.connect(self._on_region_spins)
            sp.setMinimumWidth(60)
            sp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            row2.addWidget(QLabel(name))
            row2.addWidget(sp, 1)
            self.region_spins.append(sp)
        row2.addWidget(newButton("Clear", slot=lambda: self.region_canvas.clear(),
                                 tip="Clear the region: every image size takes part again (Esc / right-click on the "
                                     "image)"))
        row3 = QHBoxLayout()
        self.region_heat = QCheckBox("Heat-map")
        self.region_heat.setChecked(True)
        self.region_heat.setToolTip("Where the shapes of the other filters (size, area, label) lie on these images")
        self.region_boxes = QCheckBox("Matches")
        self.region_boxes.setChecked(True)
        self.region_boxes.setToolTip("Outline the matching shapes (up to 2,000, spread over all of them)")
        for cb in (self.region_heat, self.region_boxes):
            cb.toggled.connect(self._on_region_layers)
            row3.addWidget(cb)
        row3.addStretch(1)
        self.bg_prev = newButton("◀", slot=lambda: self._step_background(-1), tip="Previous sample image of this size")
        self.bg_next = newButton("▶", slot=lambda: self._step_background(1), tip="Next sample image of this size")
        self.bg_label = QLabel("")
        self.bg_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # never widens
        self.bg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        apply_style(self.bg_label, "color:@muted;")
        row3.addWidget(self.bg_prev)
        row3.addWidget(self.bg_label, 2)
        row3.addWidget(self.bg_next)
        self.region_canvas = RegionCanvas()
        self.region_canvas.regionChanging.connect(self._on_region_changing)
        self.region_canvas.regionChanged.connect(lambda _r: self._region_commit())
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addLayout(row3)
        lay.addWidget(self.region_canvas, 1)
        self.region_canvas.setMinimumSize(240, 180)
        self._regions: dict = {}                   # (W, H) -> last region drawn on that size
        self._region_cache = None                  # (size, group indices, px1, py1, px2, py2, iw, ih)
        self._heat_key = None
        self._bg_keys, self._bg_i = [], 0
        self._live_timer = QTimer(self, singleShot=True, interval=30, timeout=self._region_live_count)
        return w

    def region_size(self):
        return self.region_size_box.currentData() if hasattr(self, "region_size_box") else None

    def region_rect(self):
        if not hasattr(self, "region_canvas") or self.region_size() is None:
            return None
        return self.region_canvas.region

    def _fill_sizes(self):
        d = self.data
        box = self.region_size_box
        cur = box.currentData()
        box.blockSignals(True)
        box.clear()
        for W, H, files, shapes in d.get("groups", []):
            box.addItem(f"{W} × {H}  —  {files:,} image(s) · {shapes:,} shape(s)", (W, H))
        if not box.count():
            box.addItem("No image sizes known", None)
        want = cur
        if want is None and self.app.current_key in d["fid_of"]:      # the size of the image open in the viewer
            k = self.app.current_key
            i = next((j for j, key in enumerate(d["key"]) if key == k), None)
            if i is not None:
                want = (int(d["iw"][i]), int(d["ih"][i]))
        j = box.findData(want) if want is not None else -1
        box.setCurrentIndex(j if j >= 0 else 0)
        box.blockSignals(False)
        self._region_cache = None
        self._on_region_size(apply=False)

    def _group(self):
        """Indices + float32 boxes of the shapes of the chosen image size (cached per size)."""
        size = self.region_size()
        d = self.data
        if size is None or d is None:
            return None
        c = self._region_cache
        if c is None or c[0] != size or len(c[1]) and c[1][-1] >= len(d["w"]):
            from AIO_QC.core.region import in_size
            gi = np.nonzero(in_size(d["iw"], d["ih"], size))[0]
            c = self._region_cache = (size, gi, d["px1"][gi], d["py1"][gi], d["px2"][gi], d["py2"][gi],
                                      d["iw"][gi], d["ih"][gi])
        return c

    def _region_mask(self, rect=None) -> np.ndarray:
        from AIO_QC.core.region import region_mask
        d = self.data
        m = np.zeros(len(d["w"]), bool)
        g = self._group()
        rect = rect or self.region_rect()
        if g is None or rect is None:
            return m
        size, gi, x1, y1, x2, y2, iw, ih = g
        m[gi] = region_mask(x1, y1, x2, y2, iw, ih, size, rect, self.region_mode.currentData(),
                            self.region_invert.isChecked())
        return m

    def _on_region_size(self, apply: bool = True):
        size = self.region_size()
        cv = self.region_canvas
        if size is None:
            cv.set_frame(0, 0)
            cv.empty_text = "Scan first — then pick an image size" if self.data is None else \
                "No image sizes could be read for these files"
            return
        cv.set_frame(*size)
        cv.set_region(self._regions.get(size))
        self._sync_region_spins()
        self._heat_key = None
        self._bg_keys = list((self.data or {}).get("size_keys", {}).get(size, []))
        cur = self.app.current_key
        self._bg_i = self._bg_keys.index(cur) if cur in self._bg_keys else 0
        self._load_background()
        if apply:
            self._region_commit()

    def _step_background(self, step):
        if self._bg_keys:
            self._bg_i = (self._bg_i + step) % len(self._bg_keys)
            self._load_background()

    def _load_background(self):
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QImageReader
        img = None
        if self._bg_keys:
            key = self._bg_keys[self._bg_i]
            s = self.session.by_key.get(key)
            path = s.image_path if s else None
            try:
                if path is None and s is not None:
                    path = self.session.adapter.load(s, light=True).image_path
                if path is not None:
                    r = QImageReader(str(path))
                    r.setAutoTransform(True)
                    sz = r.size()
                    if sz.isValid() and max(sz.width(), sz.height()) > 1600:
                        k = 1600 / max(sz.width(), sz.height())
                        r.setScaledSize(QSize(round(sz.width() * k), round(sz.height() * k)))
                    img = r.read()
                elif s is not None:                      # LabelMe JSON with only the embedded image
                    from .workers import load_qimage
                    ann = self.session.adapter.load(s)
                    img = load_qimage(None, ann.image_data)
            except Exception:
                img = None
            self.bg_label.setText(f"{self._bg_i + 1}/{len(self._bg_keys)} · {s.name if s else ''}")
            self.bg_label.setToolTip(s.display if s else key)
        else:
            self.bg_label.setText("")
        self.region_canvas.set_background(img)
        for b in (self.bg_prev, self.bg_next):
            b.setEnabled(len(self._bg_keys) > 1)

    def _sync_region_spins(self):
        r = self.region_canvas.region
        size = self.region_size() or (0, 0)
        vals = (r[0], r[1], r[2] - r[0], r[3] - r[1]) if r else (0, 0, 0, 0)
        for sp, v, hi in zip(self.region_spins, vals, (size[0], size[1], size[0], size[1])):
            sp.blockSignals(True)
            sp.setMaximum(max(int(hi), 0))
            sp.setValue(int(round(v)))
            sp.blockSignals(False)

    def _on_region_spins(self):
        size = self.region_size()
        if size is None:
            return
        from AIO_QC.core.region import clamp_rect
        x, y, w, h = (sp.value() for sp in self.region_spins)
        r = clamp_rect((x, y, x + w, y + h), *size) if w and h else None
        if r != self.region_canvas.region:
            self.region_canvas.set_region(r)
            self._region_commit()

    def _on_region_changing(self, rect):
        """While dragging: only the count (fast) — the table, crops and scatter follow on release."""
        self._sync_region_spins()
        self._live_timer.start()

    def _region_live_count(self):
        if self.data is None or self.region_rect() is None:
            return
        base = getattr(self, "_base_mask", None)
        m = self._region_mask()
        n = int((m & base).sum()) if base is not None and len(base) == len(m) else int(m.sum())
        self.region_canvas.count_text = f"{n:,} shape(s)"
        self.region_canvas.update()

    def _region_commit(self):
        size = self.region_size()
        if size is not None:
            self._regions[size] = self.region_canvas.region
        self._sync_region_spins()
        if self.data is None:
            return
        self._update_scatter()
        self._apply_ranges()

    def _update_scatter(self):
        d = self.data
        if d is None:
            return
        m = d["alive"].copy() if "alive" in d else np.ones(len(d["w"]), bool)
        if self.region_rect() is not None:
            m &= self._region_mask()               # the size scatter then shows just the region's shapes
        self.scatter.set_data(d["w"][m], d["h"][m], "%" if d["pct"] else "px")
        self.scatter.set_selection(*self._limits()["Width"], *self._limits()["Height"])

    def _on_region_layers(self, *_):
        cv = self.region_canvas
        cv.show_heat, cv.show_boxes = self.region_heat.isChecked(), self.region_boxes.isChecked()
        cv.update()

    def _region_visuals(self, m: np.ndarray):
        """Heat-map (shapes of the other filters on this image size), the matching boxes, the count."""
        g = self._group()
        cv = self.region_canvas
        if g is None:
            return
        d = self.data
        size, gi = g[0], g[1]
        base = self._selection_mask(region=False)
        self._base_mask = base
        key = (size, len(d["w"]), int(base.sum()), hash(base[gi].tobytes()) if len(gi) < 2_000_000 else 0)
        if key != self._heat_key:
            from AIO_QC.core.region import coverage
            sel = base[gi]
            if sel.sum() > 400_000:                # a heat-map does not need every box: an even sample
                keep = np.nonzero(sel)[0]
                sel = np.zeros(len(gi), bool)
                sel[keep[np.linspace(0, len(keep) - 1, 400_000).astype(np.int64)]] = True
            cv.set_heat(coverage(g[2], g[3], g[4], g[5], sel, *size))
            self._heat_key = key
        hit = np.nonzero(m[gi])[0]
        if len(hit) > 2000:
            hit = hit[np.linspace(0, len(hit) - 1, 2000).astype(np.int64)]
        fallback = theme_color("info")
        boxes = []
        for j in hit:
            i = gi[j]
            boxes.append((float(d["px1"][i]), float(d["py1"][i]), float(d["px2"][i]), float(d["py2"][i]),
                          self.app.label_colors.get(d["label"][i], fallback)))
        cv.set_boxes(boxes)
        if cv.region is not None:
            n_size = int(m[gi].sum())
            cv.count_text = f"{n_size:,} shape(s)"
        else:
            cv.count_text = ""
        cv.update()

    # ------------------------------------------------------------------ scan: measure everything
    def make_context(self):
        ctx = {"adapter": self.session.adapter, "pct": self.unit.currentIndex() == 1,
               "fid_of": {}, "size_keys": defaultdict(list)}
        for f in self.LIST_FIELDS + tuple(self.ARRAY_FIELDS):
            ctx[f] = []
        return ctx

    def _measure(self, ctx, s, out):
        ad = ctx["adapter"]
        ann = ad.load(s, light=True)
        if not ann.shapes:
            return
        try:                                       # exact pixel size of THIS image (Region view)
            psize = ad.image_size(s, ann)
        except Exception:
            psize = None
        pw, ph = (int(psize[0]), int(psize[1])) if psize else (0, 0)
        fid = ctx["fid_of"].setdefault(s.key, len(ctx["fid_of"]))
        if pw and ph:
            ex = ctx["size_keys"][(pw, ph)]
            if len(ex) < 60 and s.key not in ex:
                ex.append(s.key)                   # sample backgrounds for that size
        bk = (pw, ph) if ann.normalized else (1.0, 1.0)
        if ctx["pct"]:
            W, H = (1.0, 1.0) if ann.normalized else (ann.width, ann.height)
            if not (W and H):
                W, H = ad.image_size(s, ann) or (None, None)
            if not (W and H):
                raise ValueError("image size unknown")
            k = 100.0
        else:
            size = None
            if ann.normalized:
                size = ad.image_size(s, ann)
                if not size:
                    raise ValueError("image size unknown")
            W, H = (1.0 / size[0], 1.0 / size[1]) if size else (1.0, 1.0)     # normalised -> pixels
            k = 1.0
        for i, sh in enumerate(ann.shapes):
            m = shape_metrics(sh)
            if m is None:
                continue
            poly = to_polygon(sh)
            pts = poly if poly is not None else np.asarray(sh.get("points", []), float)
            out["key"].append(s.key)
            out["idx"].append(i)
            out["label"].append(str(sh.get("label", "")))
            out["w"].append(m[0] / W * k)
            out["h"].append(m[1] / H * k)
            out["a"].append(m[2] / (W * H) * (k if k == 1.0 else 100.0))
            out["box"].append((float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())))
            out["norm"].append(bool(ann.normalized))
            out["fp"].append(fingerprint(sh))             # re-identifies the shape later, even if indices shift
            b = out["box"][-1]
            out["px1"].append(b[0] * bk[0])
            out["py1"].append(b[1] * bk[1])
            out["px2"].append(b[2] * bk[0])
            out["py2"].append(b[3] * bk[1])
            out["iw"].append(pw)
            out["ih"].append(ph)
            out["fid"].append(fid)

    def scan_sample(self, ctx, s):
        self._measure(ctx, s, ctx)
        return []

    @staticmethod
    def _encode_labels(ctx):
        """Integer label codes: the label filter becomes one vectorised comparison (fast at millions of shapes)."""
        names = sorted(set(ctx["label"]))
        code = {n: i for i, n in enumerate(names)}
        ctx["label_names"] = names
        ctx["label_code"] = np.fromiter((code[l] for l in ctx["label"]), dtype=np.int32, count=len(ctx["label"]))

    def finalize(self, ctx):
        for k, dt in self.ARRAY_FIELDS.items():
            ctx[k] = np.asarray(ctx[k], dtype=dt)
        self._encode_labels(ctx)
        from AIO_QC.core.region import size_groups
        ctx["groups"] = size_groups(ctx["iw"], ctx["ih"], ctx["fid"])
        return []

    def on_scan_complete(self, ctx):
        self.data = ctx
        self._fill_sizes()
        self._update_scatter()
        self._on_spins(apply_now=True)

    # ------------------------------------------------------------------ interaction
    def _limits(self):
        out = {}
        for name, (lo, hi) in self.ranges.items():
            out[name] = (lo.value(), hi.value() or math.inf)
        return out

    def _set_spin(self, name, lo_v, hi_v):
        lo, hi = self.ranges[name]
        for sp, v in ((lo, lo_v), (hi, 0.0 if math.isinf(hi_v) else hi_v)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def _on_brush(self, wmin, wmax, hmin, hmax):
        self._set_spin("Width", wmin, wmax)
        self._set_spin("Height", hmin, hmax)
        self._apply_ranges()

    def _on_spins(self, apply_now=False):
        lim = self._limits()
        self.scatter.set_selection(lim["Width"][0], lim["Width"][1], lim["Height"][0], lim["Height"][1])
        if apply_now:
            self._apply_ranges()
        else:
            self._apply_timer.start()

    def _clear_limits(self):
        for name in self.ranges:
            self._set_spin(name, 0.0, math.inf)
        self._on_spins(apply_now=True)

    def _selection_mask(self, region: bool = True):
        """Every measured shape inside the rectangle + Area limits + Label (+ image region) — all of them, not only
        the shown rows. region=False: everything but the region (heat-map of where the other filters' shapes lie)."""
        d = self.data
        lim = self._limits()
        m = np.ones(len(d["w"]), bool)
        for name, arr in (("Width", d["w"]), ("Height", d["h"]), ("Area", d["a"])):
            lo, hi = lim[name]
            m &= (arr >= lo) & (arr <= hi)
        text = self.label.currentText().strip()
        want = None if text in ("", self.label.itemText(0)) else text
        if want is not None:
            m &= d["label_code"] == (d["label_names"].index(want) if want in d["label_names"] else -1)
        if "alive" in d:
            m &= d["alive"]
        if region and self.region_rect() is not None:
            m &= self._region_mask()
        return m

    def _selection_text(self) -> str:
        lim, unit = self._limits(), "%" if self.data["pct"] else "px"
        parts = []
        for name in ("Width", "Height", "Area"):
            lo, hi = lim[name]
            if lo > 0 or not math.isinf(hi):
                parts.append(f"{name.lower()} {lo:.3g}–{'∞' if math.isinf(hi) else f'{hi:.3g}'}")
        lab = self.label.currentText().strip()
        if lab and lab != self.label.itemText(0):
            parts.append(f"label {lab}")
        text = (", ".join(parts) or "no limits") + f" ({unit})"
        r = self.region_rect()
        if r is not None:
            from AIO_QC.core.region import MODES
            W, H = self.region_size()
            text += (f"; {'outside' if self.region_invert.isChecked() else 'in'} region x {r[0]:.0f}–{r[2]:.0f}, "
                     f"y {r[1]:.0f}–{r[3]:.0f} of {W}×{H} images ({MODES[self.region_mode.currentData()].lower()})")
        return text

    def _apply_ranges(self):
        d = self.data
        if d is None or not len(d["w"]):
            return
        m = self._selection_mask()
        idx = np.nonzero(m)[0]
        n = len(idx)
        shown = idx[np.argsort(d["a"][idx], kind="stable")][: self.MAX_ROWS]      # smallest first: junk surfaces first
        rows = [ResultRow(d["key"][i], (d["label"][i], d["idx"][i], round(float(d["w"][i]), 3),
                                        round(float(d["h"][i]), 3), round(float(d["a"][i]), 3)), d["idx"][i],
                          payload=int(i)) for i in shown]
        self.table.setRowCount(0)
        self.rows = []
        self._append_rows(rows)
        self._rebuild_filter()
        self.scatter.count_text = f"{n:,} shapes" + (f" (showing {len(shown):,})" if n > len(shown) else "")
        self.scatter.update()
        self._refresh_cards()
        self._match_text = (f"{n:,} of {len(d['w']):,} shapes match" +
                            (f" — table and crops show the {len(shown):,} smallest; narrow the selection to see all"
                             if n > len(shown) else ""))
        self.summary.setText(self._match_text)
        self._region_visuals(m)

    def _refresh_cards(self):
        d = self.data
        if d is None:
            return
        self.prune_marks()
        unit = "%" if d["pct"] else "px"
        cards = []
        for row in self.visible_rows()[:3000]:
            i = row.payload
            b = d["box"][i]
            col = self.app.label_colors.get(d["label"][i])
            s = self.session.by_key.get(row.key)
            cards.append(Card(row, [Panel(row.key, b, [Shape(box_points(*b), col.name() if col else hexc("info"))],
                                          d["norm"][i])],
                              chips=[(f"{d['w'][i]:.3g}×{d['h'][i]:.3g} {unit}", "info"),
                                     (f"area {d['a'][i]:.3g}", "muted")],
                              lines=[d["label"][i], s.display if s else row.key]))
        self.cards.set_cards(cards)

    # ------------------------------------------------------------------ act on the selection
    def _targets(self, indices) -> dict:
        d = self.data
        out = defaultdict(list)
        for i in indices:
            out[d["key"][i]].append((int(d["idx"][i]), d["fp"][i]))
        return out

    def _act_on_selection(self):
        d = self.data
        if d is None or not len(d["w"]):
            QMessageBox.information(self, self.TITLE, "Scan first, then drag a rectangle on the scatter or the region "
                                                      "view.")
            return
        idx = np.nonzero(self._selection_mask())[0]
        if not len(idx):
            QMessageBox.information(self, self.TITLE, "The selection is empty.")
            return
        lim = self._limits()
        if self.region_rect() is None and all(lim[n][0] <= 0 and math.isinf(lim[n][1]) for n in lim) and QMessageBox.question(
                self, self.TITLE, f"No limits are set — the selection is EVERY measured shape ({len(idx):,}).\n"
                                  "Continue?") != QMessageBox.StandardButton.Yes:
            return
        labels = sorted(set(self.session.label_totals()) | set(self.session.adapter.class_names))
        dlg = SelectionActionDialog(self, idx, self._selection_text(), labels)
        if not dlg.exec():
            return
        op, new_label, flag = dlg.result_op()
        self._run_op(self._targets(idx), op, new_label, flag, self._selection_text())

    def _delete_ticked(self):
        rows = [r for r in self.target_rows() if r.payload is not None]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows first — or use “Selection…” to act on "
                                                      "everything inside the rectangle.")
            return
        if not self.confirm(f"Delete {len(rows):,} ticked shape(s) from {len({r.key for r in rows}):,} file(s)?\n\n"
                            "The originals are backed up (Undo)."):
            return
        self._run_op(self._targets([r.payload for r in rows]), "delete", None, False, "ticked rows")

    def _run_op(self, targets, op, new_label, flag, where):
        n = sum(len(v) for v in targets.values())
        if op == "flag":
            keys = list(targets)
            self.session.set_review(keys, REVIEW_FLAG, f"Dim / Area: {n} shape(s) — {where}")
            logger.info(f"[Dim/Area] flagged {len(keys)} file(s) with {n} shape(s) in the selection ({where})")
            self.report("Done", [f"Flagged {len(keys):,} file(s)."], [])
            return
        st = self.session.apply_to_shapes(self, targets, op, new_label, what="dimarea")
        if st["files"]:
            self._last_backup = st["backup"]
            self.undo_btn.setEnabled(True)
            self.undo_btn.setToolTip(f"Put back the {st['files']:,} file(s) changed by the last "
                                     f"{'deletion' if op == 'delete' else 'relabel'} (backup: {st['backup'].label})")
        if flag and st["changed"]:
            self.session.set_review(st["changed"], REVIEW_FLAG,
                                    f"Dim / Area: {'deleted' if op == 'delete' else 'relabelled'} shapes — {where}")
        self.after_edit(list(targets))
        verb = "Deleted" if op == "delete" else f"Relabelled to “{new_label}”"
        lines = [f"{verb}: {st['done']:,} of {n:,} shape(s) in {st['files']:,} file(s)."]
        if st["missing"]:
            lines.append(f"{st['missing']:,} shape(s) had changed since the scan (edited / removed meanwhile) and were "
                         "left alone.")
        if st["files"]:
            lines.append(f"Originals backed up ({len(st['backup'].saved):,} file(s)) — “↶ Undo” puts them back.")
        logger.info(f"[Dim/Area] {verb.lower()} {st['done']} shape(s) in {st['files']} file(s) ({where}); "
                    f"{st['missing']} changed since the scan; {len(st['errors'])} error(s); backup: {st['backup'].label}")
        self.report("Done", lines, st["errors"])

    def _undo(self):
        b = self._last_backup
        if b is None:
            return
        if not self.confirm(f"Put back the original version of {len(b.after):,} file(s)?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.after_edit(keys)
        logger.info(f"[Dim/Area] undo: {len(keys)} file(s) restored, {len(skipped)} skipped (changed since)")
        self.report("Undone", [f"Restored {len(keys):,} file(s)."] +
                    ([f"{len(skipped):,} file(s) were changed after the operation and were left as they are."]
                     if skipped else []), [])

    def after_edit(self, keys):
        """Shapes were deleted: drop those files' measurements, re-measure just them, re-apply the selection."""
        d = self.data
        if d is None:
            return super().after_edit(keys)
        ks = set(keys)
        alive = d.get("alive", np.ones(len(d["w"]), bool))
        alive &= np.fromiter((k not in ks for k in d["key"]), bool, count=len(d["key"]))
        extra = {f: [] for f in self.LIST_FIELDS + tuple(self.ARRAY_FIELDS)}
        extra.update(pct=d["pct"], adapter=d["adapter"], fid_of=d["fid_of"], size_keys=d["size_keys"])
        for k in ks:
            s = self.session.by_key.get(k)
            if s is not None:
                try:
                    self._measure(extra, s, extra)
                except Exception as exc:
                    self.errors.append(f"{s.name}: {exc}")
        for f in self.LIST_FIELDS:
            d[f] = list(d[f]) + extra[f]
        for f, dt in self.ARRAY_FIELDS.items():
            d[f] = np.concatenate([d[f], np.asarray(extra[f], dtype=dt)])
        d["alive"] = np.concatenate([alive, np.ones(len(extra["w"]), bool)])
        self._encode_labels(d)
        self._region_cache = None
        self._update_scatter()
        self._on_spins(apply_now=True)

    def summarize(self):
        return getattr(self, "_match_text", "") or "No shapes"


class SelectionActionDialog(QDialog):
    """What to do with every shape in the Dim / Area selection: delete · relabel · only flag the files."""

    def __init__(self, parent, indices, selection_text: str, labels: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Act on the selection")
        d = parent.data
        keys = {d["key"][i] for i in indices}
        locked = [k for k in keys if k in parent.session.locked]
        per = Counter(d["label"][i] for i in indices)
        head = QLabel(f"<b>{len(indices):,} shape(s)</b> in <b>{len(keys):,}</b> file(s) — {selection_text}")
        head.setTextFormat(Qt.TextFormat.RichText)
        head.setWordWrap(True)
        top = ", ".join(f"{lab} × {n:,}" for lab, n in per.most_common(8)) + (" …" if len(per) > 8 else "")
        by = QLabel(top)
        by.setWordWrap(True)
        apply_style(by, "color:@muted;")
        self.rb_del = QRadioButton("Delete these shapes")
        self.rb_rel = QRadioButton("Relabel them to:")
        self.rb_flag = QRadioButton("Only flag the files for review (nothing is changed)")
        self.rb_del.setChecked(True)
        self.new_label = QComboBox()
        self.new_label.setEditable(True)
        self.new_label.addItems(labels)
        self.new_label.setEnabled(False)
        self.rb_rel.toggled.connect(self.new_label.setEnabled)
        rel = QHBoxLayout()
        rel.addWidget(self.rb_rel)
        rel.addWidget(self.new_label, 1)
        self.flag = QCheckBox("Also flag the changed files for review")
        self.rb_flag.toggled.connect(lambda on: self.flag.setEnabled(not on))
        note = QLabel("Each shape is re-identified in the file as it is now — shapes edited or removed since the scan "
                      "are left alone. The originals are backed up and “↶ Undo” puts them back."
                      + (f"<br><span style='color:{hexc('warn')}'>{len(locked):,} file(s) are open in a Fix-in-Editor "
                         "session and will be skipped.</span>" if locked else ""))
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setWordWrap(True)
        apply_style(note, "color:@muted;")
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        for w in (head, by):
            lay.addWidget(w)
        lay.addSpacing(6)
        lay.addWidget(self.rb_del)
        lay.addLayout(rel)
        lay.addWidget(self.rb_flag)
        lay.addWidget(self.flag)
        lay.addWidget(note)
        lay.addWidget(bb)
        self.setMinimumWidth(520)

    def _accept(self):
        if self.rb_rel.isChecked() and not self.new_label.currentText().strip():
            return
        self.accept()

    def result_op(self):
        if self.rb_flag.isChecked():
            return "flag", None, False
        if self.rb_rel.isChecked():
            return "relabel", self.new_label.currentText().strip(), self.flag.isChecked()
        return "delete", None, self.flag.isChecked()


# =============================================================================
# 🚨 Outliers
# =============================================================================
METRICS = ["area", "width", "height", "aspect ratio (w/h)"]


class OutlierDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Outliers", "🚨"
    HELP = ("Per-class IQR on (log) shape size relative to the image, so mixed resolutions are comparable. "
            "Flags shapes far outside their class's typical range — often wrong labels, merged/split boxes or "
            "stray clicks. Classes with fewer samples than the minimum are skipped.")
    COLUMNS = ["Label", "Shape", "Value", "Class median", "Low bound", "High bound", "Direction"]
    FILTER_COLUMN = 0

    def build_params(self, form):
        self.metric = QComboBox()
        self.metric.addItems(METRICS)
        self.k = dspin(0.5, 10, 1.5, 1, 0.5)
        self.log = QCheckBox("Log scale (recommended: sizes are heavy-tailed)")
        self.log.setChecked(True)
        self.per_class = QCheckBox("Per class")
        self.per_class.setChecked(True)
        self.min_n = ispin(4, 100000, 8)
        form.addRow("Metric:", self.metric)
        form.addRow("IQR multiplier k:", self.k)
        form.addRow("Min samples per class:", self.min_n)
        form.addRow(self.log)
        form.addRow(self.per_class)
        self.add_action("🗑 Delete shapes", lambda: self.delete_shape_rows(self.target_rows(), "outlier shape(s)"))

    def make_context(self):
        return {"adapter": self.session.adapter, "metric": self.metric.currentText(), "k": self.k.value(),
                "log": self.log.isChecked(), "per_class": self.per_class.isChecked(), "min_n": self.min_n.value(),
                "values": defaultdict(list), "bounds": {}}

    def _measure(self, ctx, s):
        ad = ctx["adapter"]
        ann = ad.load(s, light=True)
        W, H = (None, None) if ann.normalized else (ann.width, ann.height)
        if not ann.normalized and not (W and H):
            W, H = ad.image_size(s, ann) or (None, None)
        out = []
        for i, sh in enumerate(ann.shapes):
            v = shape_measure(sh, ctx["metric"], W, H)
            if v is not None and v > 0:
                out.append((i, str(sh.get("label")), v))
        return out

    def _t(self, ctx, v):
        return math.log(v) if ctx["log"] else v

    def _inv(self, ctx, v):
        return math.exp(v) if ctx["log"] else v

    def scan_sample(self, ctx, s):
        for i, label, v in self._measure(ctx, s):
            ctx["values"][label if ctx["per_class"] else "*"].append((s.key, i, label, v))
        return []

    def _fmt(self, ctx, v):
        return round(v * 100, 4) if ctx["metric"] != METRICS[3] else round(v, 3)

    def _check(self, ctx, key, i, label, v):
        b = ctx["bounds"].get(label if ctx["per_class"] else "*")
        if b is None:
            return None
        lo, hi, med = b
        t = self._t(ctx, v)
        if lo <= t <= hi:
            return None
        return ResultRow(key, (label, i, self._fmt(ctx, v), self._fmt(ctx, self._inv(ctx, med)),
                               self._fmt(ctx, self._inv(ctx, lo)), self._fmt(ctx, self._inv(ctx, hi)),
                               "too small" if t < lo else "too large"), i,
                         payload=t, tone="warning" if t < lo else "error")

    def finalize(self, ctx):
        for group, vals in ctx["values"].items():
            if len(vals) >= ctx["min_n"]:
                ctx["bounds"][group] = iqr_bounds([self._t(ctx, v) for *_, v in vals], ctx["k"])
        rows = []
        for vals in ctx["values"].values():
            for key, i, label, v in vals:
                r = self._check(ctx, key, i, label, v)
                if r:
                    rows.append(r)
        # keep compact per-class distributions (float32, chart space) for the strip chart; drop the per-shape tuples
        ctx["dist"] = {g: np.fromiter((self._t(ctx, v) for *_, v in vals), dtype=np.float32, count=len(vals))
                       for g, vals in ctx["values"].items()}
        ctx["values"] = defaultdict(list)
        return rows

    def rescan_sample(self, ctx, s):
        return [r for i, label, v in self._measure(ctx, s) if (r := self._check(ctx, s.key, i, label, v))]

    def columns_for(self, ctx):
        unit = "" if ctx["metric"] == METRICS[3] else " (% img)"
        return ["Label", "Shape", f"Value{unit}", "Class median", "Low bound", "High bound", "Direction"]

    def post_init(self):
        """Chart tab (visual, default) + Table tab."""
        from PyQt6.QtWidgets import QScrollArea
        self.init_marks()
        self.chart = StripChart()
        self.chart.activated.connect(self.on_row_activated)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.chart)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.chart_hint = QLabel("Each row is one class: grey = where its shapes are (darker = more), green band = the "
                                 "normal range, black tick = median. Red dots are unusually large, amber unusually "
                                 "small — click a dot to open that shape.")
        self.chart_hint.setWordWrap(True)
        apply_style(self.chart_hint, "color:@muted;")
        lay.addWidget(self.chart_hint)
        lay.addWidget(area, 1)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(w, "Chart")
        self.tabs.addTab(self.table, "Table")
        self.table_host.addWidget(self.tabs)

    def _refresh_chart(self):
        if not hasattr(self, "chart") or not self.ctx or "dist" not in self.ctx:
            return
        ctx = self.ctx
        by_group = defaultdict(list)
        for row in self.rows:
            if row is not None and row.payload is not None:
                g = row.cells[0] if ctx["per_class"] else "*"
                by_group[g].append((float(row.payload), row, "small" if row.cells[6] == "too small" else "large"))
        groups = []
        for g, arr in sorted(ctx["dist"].items(), key=lambda kv: -len(kv[1])):
            b = ctx["bounds"].get(g)
            if b is None or not len(arr):
                continue
            groups.append({"label": "all classes" if g == "*" else g, "values": arr, "lo": b[0], "hi": b[1],
                           "med": b[2], "outliers": by_group.get(g, [])})
        pct = ctx["metric"] != METRICS[3]

        def fmt(v):
            raw = self._inv(ctx, v)
            return f"{raw * 100:.3g}%" if pct else f"{raw:.3g}"
        self.chart.set_data(groups, fmt, ctx["log"])
        skipped = len(ctx["dist"]) - len(groups)
        self.tabs.setTabText(0, f"Chart ({len(groups)} classes" + (f", {skipped} too small" if skipped else "") + ")")

    def summarize(self):
        self._refresh_chart()
        return f"{len([r for r in self.rows if r]):,} outlier(s) across {len(self.ctx.get('bounds', {}))} class(es)"


# =============================================================================
# 🧠 Smart filter
# =============================================================================
class SmartFilterDialog(BaseScanDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Smart Filter", "🧠"
    HELP = ("File-level query: files containing any/all of the included labels, none of the excluded ones, "
            "and an object count in range. Push the result to the file list to review one by one.")
    COLUMNS = ["Objects", "Labels"]
    CHECKABLE = False
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        labels = [l for l, _ in self.session.label_totals().most_common()]

        def checklist():
            w = QListWidget()
            w.setMaximumHeight(130)
            for l in labels:
                it = QListWidgetItem(l)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked)
                w.addItem(it)
            return w

        self.include, self.exclude = checklist(), checklist()
        self.inc_mode = QComboBox()
        self.inc_mode.addItems(["any of", "all of"])
        lists = QWidget()
        h = QHBoxLayout(lists)
        h.setContentsMargins(0, 0, 0, 0)
        for title, w, extra in (("Include", self.include, self.inc_mode), ("Exclude", self.exclude, None)):
            v = QVBoxLayout()
            head = QHBoxLayout()
            head.addWidget(QLabel(title))
            if extra is not None:
                head.addWidget(extra)
            head.addStretch(1)
            v.addLayout(head)
            v.addWidget(w)
            h.addLayout(v)
        self.min_obj = ispin(0, 100000, 0)
        self.max_obj = ispin(0, 100000, 0)
        self.max_obj.setSpecialValueText(NO_LIMIT)
        rng = QWidget()
        r = QHBoxLayout(rng)
        r.setContentsMargins(0, 0, 0, 0)
        for wdg in (QLabel("min"), self.min_obj, QLabel("max"), self.max_obj):
            r.addWidget(wdg)
        r.addStretch(1)
        self.split = QComboBox()
        self.split.addItem("(all)", "")
        for sp in self.session.splits:
            self.split.addItem(sp, sp)
        form.addRow(lists)
        form.addRow("Objects per file:", rng)
        if self.session.splits:
            form.addRow("Split:", self.split)
        if not labels:
            form.addRow(QLabel("Label index still building — reopen in a moment for the label lists."))

    @staticmethod
    def _checked(w):
        return {w.item(i).text() for i in range(w.count()) if w.item(i).checkState() == Qt.CheckState.Checked}

    def make_context(self):
        return {"adapter": self.session.adapter, "inc": self._checked(self.include),
                "all": self.inc_mode.currentIndex() == 1, "exc": self._checked(self.exclude),
                "min": self.min_obj.value(), "max": self.max_obj.value() or math.inf,
                "split": self.split.currentData() or ""}

    def scan_sample(self, ctx, s):
        if ctx["split"] and s.split != ctx["split"]:
            return []
        counts = ctx["adapter"].load(s, light=True).label_counts()
        n = sum(counts.values())
        present = set(counts)
        if not ctx["min"] <= n <= ctx["max"]:
            return []
        if ctx["inc"]:
            ok = ctx["inc"] <= present if ctx["all"] else bool(ctx["inc"] & present)
            if not ok:
                return []
        if ctx["exc"] & present:
            return []
        return [ResultRow(s.key, (n, ", ".join(f"{l}×{c}" for l, c in counts.most_common())))]

    def summarize(self):
        return f"{len([r for r in self.rows if r]):,} matching file(s) — use 👁 Show in file list to review"
