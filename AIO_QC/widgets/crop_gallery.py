# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🧩 Crop gallery: every instance of one label as a grid of crops.

Wrong labels, junk boxes and duplicates jump out when you see 500 crops of "hsrp_white" side by side.

Built for millions of instances:
  * instances live in a column store (core.instances.InstanceStore: ~40 bytes + outline each, no Python object per
    instance); the view model is a numpy array of ids, so sorting 1M instances is one argsort, not a Python sort
  * collection skips files without the label (label index), uses pure-Python geometry and — for big datasets — a
    process pool; tiles appear while it runs
  * the grid is a uniform-size list with a custom tile painter (no per-item layout objects), selections are
    ranges (Ctrl+A on a million crops is instant), counts are kept running
  * crops are decoded lazily — only for tiles on screen — newest request first, one decode per image for all its
    crops (JPEG scaled decode), region-only decode for single crops; a bounded LRU keeps memory flat
  * edits made anywhere (this window, other tools, other programs) refresh just the touched files in place
"""
from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import (
    QAbstractListModel, QModelIndex, QObject, QPointF, QRect, QRunnable, QSize, Qt, QThread, QThreadPool, QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QImage, QImageIOHandler, QImageReader, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel, QListView, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QStyle, QStyledItemDelegate, QVBoxLayout,
)

from AIO_QC.core.dataset import DETECTION_TYPES  # noqa: F401  (kept for callers that import it from here)
from AIO_QC.core.instances import SORTS, Instance, InstanceStore, collect_file, iter_collect, shuffle_seed
from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.utils import LRUCache, newButton, style as apply_style, theme_color

__all__ = ["CropGalleryDialog", "Instance", "SORTS", "crop_geometry"]
LOOK_SORTS = ["Least typical first", "Grouped by look"]
LOOK_TIPS = ["Crops that look least like the rest of this label first (needs a Visual Similarity run)",
             "Look-alike crops next to each other: odd ones stand out at group borders (needs a Visual Similarity run)"]


# =============================================================================
# Collection (worker thread; a process pool inside for big datasets)
# =============================================================================
class CollectWorker(QThread):
    progress = pyqtSignal(int, int)
    batch = pyqtSignal(list)               # rows found since the last batch (tiles appear while collecting)
    done = pyqtSignal(list, list)          # [], errors

    def __init__(self, adapter, samples, label):
        super().__init__()
        self.adapter, self.samples, self.label = adapter, samples, label
        self._stop = False
        self.setObjectName("CropGalleryCollect")

    def stop(self):
        self._stop = True

    def run(self):
        errors, pending, last, first = [], [], time.monotonic(), True
        n = len(self.samples)
        for files, rows, errs in iter_collect(self.adapter, self.samples, self.label, stop=lambda: self._stop):
            pending.extend(rows)
            errors.extend(errs)
            now = time.monotonic()
            # first tiles fast (as soon as one screenful exists), then every 250 ms
            if pending and ((first and len(pending) >= 60) or now - last > 0.25):
                self.batch.emit(pending)
                pending, first, last = [], False, now
            self.progress.emit(files, n)
        if self._stop:
            return
        if pending:
            self.batch.emit(pending)
        self.progress.emit(n, n)
        self.done.emit([], errors)


def collect_instances(adapter, sample, label=None, indices=None) -> list[Instance]:
    """Compatibility helper: Instance objects of one file."""
    return [Instance(*r) for r in collect_file(adapter, sample, label, indices)]


# =============================================================================
# Crop decoding (thread pool)
# =============================================================================
class CropSignals(QObject):
    loaded = pyqtSignal(object, int, QImage)      # cache key (instance id), generation, image


def crop_geometry(inst, W, H, tile, pad):
    """(points in pixels, crop rect, scale k, output size) for one instance in a W×H image."""
    sx, sy = (W, H) if inst.normalized else (1.0, 1.0)
    pts = [(x * sx, y * sy) for x, y in inst.points]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    m = max(bw, bh) * pad
    rx1, ry1 = max(0.0, x1 - m), max(0.0, y1 - m)
    rect = QRect(int(rx1), int(ry1), max(1, int(min(W, x2 + m) - rx1)), max(1, int(min(H, y2 + m) - ry1)))
    k = min(tile / rect.width(), tile / rect.height(), 8.0)          # upscale tiny crops (max 8x)
    out = QSize(max(1, round(rect.width() * k)), max(1, round(rect.height() * k)))
    return pts, rect, k, out


def draw_outline(img: QImage, pts, rect: QRect, k: float, color: QColor) -> QImage:
    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(color, 2))
    poly = QPolygonF([QPointF((x - rect.x()) * k, (y - rect.y()) * k) for x, y in pts])
    p.drawPolygon(poly) if len(pts) > 2 else p.drawPolyline(poly)
    p.end()
    return img


class ImageBatchTask(QRunnable):
    """
    All pending crops of ONE image in one decode. With several crops the image is decoded once at the smallest
    scale that still gives every crop full tile resolution (JPEG decodes 1/2, 1/4, 1/8 scale natively, which is
    much faster than a full decode); a single crop uses the region-only decode (CropTask).
    """

    def __init__(self, signals, path, jobs, tile: int, pad: float, generation: int):
        super().__init__()
        self.signals, self.path, self.jobs, self.tile, self.pad, self.generation = signals, path, jobs, tile, pad, generation

    def run(self):
        done = set()
        try:
            if self.path is None:
                pass
            elif len(self.jobs) == 1:
                ck, inst, color = self.jobs[0]
                img = CropTask(self.signals, ck, inst, self.tile, self.pad, color, self.generation)._crop()
                self.signals.loaded.emit(ck, self.generation, img)
                done.add(ck)
            else:
                self._batch(done)
        except Exception:
            pass
        for ck, _inst, _c in self.jobs:                 # anything that failed: show the "no image" tile
            if ck not in done:
                self.signals.loaded.emit(ck, self.generation, QImage())

    def _batch(self, done):
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)
        full = reader.size()
        rotated = reader.transformation() != QImageIOHandler.Transformation.TransformationNone
        if rotated or not full.isValid():
            src = reader.read()
            if src.isNull():
                return
            W, H = src.width(), src.height()
            geo = [crop_geometry(inst, W, H, self.tile, self.pad) for _ck, inst, _c in self.jobs]
            s = 1.0
        else:
            W, H = full.width(), full.height()
            geo = [crop_geometry(inst, W, H, self.tile, self.pad) for _ck, inst, _c in self.jobs]
            s = min(1.0, max(g[2] for g in geo))       # the most demanding crop decides the decode scale
            if s < 0.9:
                reader.setScaledSize(QSize(max(1, round(W * s)), max(1, round(H * s))))
            else:
                s = 1.0
            src = reader.read()
            if src.isNull():
                return
        for (ck, _inst, color), (pts, rect, k, out) in zip(self.jobs, geo):
            rs = QRect(int(rect.x() * s), int(rect.y() * s), max(1, round(rect.width() * s)),
                       max(1, round(rect.height() * s)))
            crop = src.copy(rs).scaled(out, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.signals.loaded.emit(ck, self.generation, draw_outline(crop, pts, rect, k, color))
            done.add(ck)


class CropTask(QRunnable):
    def __init__(self, signals, ckey, inst: Instance, size: int, pad: float, color: QColor, generation: int):
        super().__init__()
        self.signals, self.ckey, self.inst, self.tile, self.pad = signals, ckey, inst, size, pad
        self.color, self.generation = color, generation

    def run(self):
        img = QImage()
        try:
            img = self._crop()
        except Exception:
            img = QImage()
        self.signals.loaded.emit(self.ckey, self.generation, img)

    def _crop(self) -> QImage:
        inst = self.inst
        if inst.image is None:
            return QImage()
        reader = QImageReader(str(inst.image))
        reader.setAutoTransform(True)
        full = reader.size()
        rotated = reader.transformation() != QImageIOHandler.Transformation.TransformationNone
        if rotated or not full.isValid():
            src = reader.read()               # EXIF-rotated: decode fully, then crop in display space
            if src.isNull():
                return QImage()
            W, H = src.width(), src.height()
        else:
            W, H = full.width(), full.height()
        pts, rect, k, out_size = crop_geometry(inst, W, H, self.tile, self.pad)
        if rotated or not full.isValid():
            crop = src.copy(rect).scaled(out_size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
        else:
            reader.setClipRect(rect)
            reader.setScaledSize(out_size)
            crop = reader.read()
            if crop.isNull():
                return QImage()
        return draw_outline(crop, pts, rect, k, self.color)


# =============================================================================
# Model: rows -> instance ids (numpy)
# =============================================================================
class InstanceModel(QAbstractListModel):
    def __init__(self, gallery):
        super().__init__(gallery)
        self.g = gallery
        self.order = np.zeros(0, dtype=np.int64)       # row -> instance id
        self.pos = np.zeros(0, dtype=np.int64)         # instance id -> row (-1 = not shown)
        self._n = 0                                    # rows in use (order has spare capacity while streaming)

    def ids(self) -> np.ndarray:
        return self.order[:self._n]

    def _rebuild_pos(self, size: int):
        self.pos = np.full(max(size, 1), -1, dtype=np.int64)
        self.pos[self.order[:self._n]] = np.arange(self._n, dtype=np.int64)

    def set_order(self, order: np.ndarray, store_size: int):
        self.beginResetModel()
        self.order = np.asarray(order, dtype=np.int64).copy()
        self._n = len(self.order)
        self._rebuild_pos(store_size)
        self.endResetModel()

    def append_ids(self, ids: np.ndarray, store_size: int):
        """Streaming: new instances go to the end (the chosen sort is applied when collection finishes)."""
        if not len(ids):
            return
        n, m = self._n, len(ids)
        if n + m > len(self.order):                    # grow geometrically: O(1) amortised per instance
            grown = np.empty(max(2 * len(self.order), n + m, 1024), dtype=np.int64)
            grown[:n] = self.order[:n]
            self.order = grown
        if store_size > len(self.pos):
            grown = np.full(max(2 * len(self.pos), store_size, 1024), -1, dtype=np.int64)
            grown[:len(self.pos)] = self.pos
            self.pos = grown
        self.beginInsertRows(QModelIndex(), n, n + m - 1)
        self.order[n:n + m] = ids
        self.pos[ids] = np.arange(n, n + m, dtype=np.int64)
        self._n = n + m
        self.endInsertRows()

    def row_of(self, iid: int) -> int:
        return int(self.pos[iid]) if 0 <= iid < len(self.pos) else -1

    def iid(self, row: int) -> int:
        return int(self.order[row])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._n

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= self._n:
            return None
        iid = int(self.order[index.row()])
        g = self.g
        if role == Qt.ItemDataRole.DecorationRole:
            return g.pixmap_for(iid)
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            st, key = g.store, g.store.key_of(iid)
            s = g.session.by_key.get(key)
            rv = g.session.review.status(key)
            if role == Qt.ItemDataRole.ToolTipRole:
                mark = {"ok": "\n✓ file reviewed OK", "flag": "\n⚑ file flagged", "fixed": "\n🔧 file fixed"}.get(rv, "")
                return f"{s.display if s else key}\nshape #{st.idx[iid]} · {st.label_of(iid)}{mark}"
            name = s.name if s else "?"
            flag = {"flag": "⚑ ", "fixed": "🔧 "}.get(rv, "")
            return flag + (name if len(name) <= 14 else name[:6] + "…" + name[-6:]) + f" #{st.idx[iid]}"
        return None


class TileDelegate(QStyledItemDelegate):
    """Paints a crop tile (pixmap + caption) directly: no per-item layout objects, constant cost per visible tile."""

    def __init__(self, gallery):
        super().__init__(gallery)
        self.g = gallery

    def sizeHint(self, option, index):
        return self.g.grid_size

    def paint(self, p: QPainter, opt, index):
        r = opt.rect
        tile = self.g.tile
        p.save()
        if opt.state & QStyle.StateFlag.State_Selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(opt.palette.highlight())
            p.drawRoundedRect(r.adjusted(2, 2, -2, -2), 6, 6)
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.setPen(QPen(opt.palette.highlight().color(), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(2, 2, -3, -3), 6, 6)
        pix = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(pix, QPixmap) and not pix.isNull():
            dpr = pix.devicePixelRatio() or 1.0
            w, h = pix.width() / dpr, pix.height() / dpr
            s = min(1.0, tile / max(w, 1), tile / max(h, 1))
            w, h = int(w * s), int(h * s)
            x = r.x() + (r.width() - w) // 2
            y = r.y() + 6 + (tile - h) // 2
            p.drawPixmap(QRect(x, y, w, h), pix)
        if self.g.show_names:
            text = index.data(Qt.ItemDataRole.DisplayRole) or ""
            tr = QRect(r.x() + 4, r.y() + 8 + tile, r.width() - 8, r.height() - tile - 10)
            role = opt.palette.ColorRole.HighlightedText if opt.state & QStyle.StateFlag.State_Selected \
                else opt.palette.ColorRole.Text
            p.setPen(opt.palette.color(role))
            p.drawText(tr, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       opt.fontMetrics.elidedText(text, Qt.TextElideMode.ElideMiddle, tr.width()))
        p.restore()


# =============================================================================
# Dialog
# =============================================================================
class CropGalleryDialog(QDialog):
    TITLE, ICON = "Crop Gallery", "🧩"
    SUPPORTED = None                     # every type with shapes; classification datasets simply have none

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(int(app.width() * 0.8), int(app.height() * 0.85))
        self.cache = LRUCache(capacity=2500, max_bytes=400 * 2 ** 20)
        self.signals = CropSignals()
        self.signals.loaded.connect(self._on_crop)
        self.generation = 0
        self._inflight: set = set()
        self._priority = 0
        self._pending: dict = {}                      # image path -> [(instance id, instance, colour)]
        self._flush = QTimer(self, singleShot=True, interval=0, timeout=self._flush_pending)
        self.worker: CollectWorker | None = None
        self.show_names = True
        self.store = InstanceStore()
        self.instance_mode = False            # True when opened on an explicit list (e.g. a confusion-matrix cell)
        self.label = ""
        self._seed = shuffle_seed()
        self._changed_keys: set = set()
        self._changed_timer = QTimer(self, singleShot=True, interval=300, timeout=self._apply_outside_changes)

        # controls ---------------------------------------------------------------
        self.label_box = QComboBox()
        totals = self.session.label_totals()
        for label, n in totals.most_common():
            self.label_box.addItem(f"{label}  ({n:,})", label)
        self.split_box = QComboBox()
        self.split_box.addItem("All splits", "")
        for sp in self.session.splits:
            self.split_box.addItem(sp, sp)
        self.split_box.setVisible(bool(self.session.splits))
        self.scope_box = QComboBox()
        self.scope_box.addItems(["All files", "Files in current list"])
        self.sort_box = QComboBox()
        self.sort_box.addItems(SORTS + LOOK_SORTS)
        for i, tip in enumerate(LOOK_TIPS, len(SORTS)):
            self.sort_box.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.sort_box.setCurrentIndex(1)
        self.sort_box.currentIndexChanged.connect(self._apply_sort)
        self.pad = QSpinBox()
        self.pad.setRange(0, 200)
        self.pad.setValue(20)
        self.pad.setSuffix(" % pad")
        self.pad.valueChanged.connect(self._invalidate)
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(64, 256)
        self.size_slider.setValue(128)
        self.size_slider.setFixedWidth(140)
        self.size_slider.sliderReleased.connect(self._resize_tiles)
        load = newButton("▶ Load", slot=lambda: self.load())
        load.setDefault(True)
        top = QHBoxLayout()
        for w in (QLabel("Label:"), self.label_box, self.split_box, self.scope_box, QLabel("Sort:"), self.sort_box,
                  self.pad, QLabel("Size:"), self.size_slider, load):
            top.addWidget(w)
        top.addStretch(1)

        # grid ----------------------------------------------------------------------
        self.model = InstanceModel(self)
        self.view = QListView()
        self.view.setViewMode(QListView.ViewMode.ListMode)          # list layout: O(1) per item, no item objects
        self.view.setFlow(QListView.Flow.LeftToRight)
        self.view.setWrapping(True)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setMovement(QListView.Movement.Static)
        self.view.setUniformItemSizes(True)
        self.view.setMouseTracking(True)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setItemDelegate(TileDelegate(self))
        self.view.setModel(self.model)
        self.view.doubleClicked.connect(self._open)
        self.view.selectionModel().selectionChanged.connect(lambda *_: self._update_status())
        self._apply_tile_size()

        # actions -------------------------------------------------------------------
        bar = QHBoxLayout()
        self.status = QLabel("Pick a label and press ▶ Load")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        bar.addWidget(self.status, 1)
        bar.addWidget(self.progress)
        for text, slot, tip in (
            ("✏️ Relabel…", self.relabel, "Change the label of the selected shapes (R)"),
            ("🗑 Delete", self.delete, "Delete the selected shapes (Del)"),
            ("⚑ Flag files", self.flag_files, "Flag the files of the selected crops for review"),
            ("👁 Show files", self.show_files, "Restrict the main file list to the selected crops' files"),
            ("📘 ✓ Example", lambda: self.add_example("good"),
             "Add the selected crops as ✓ good examples to their class's guideline (Labels ▸ Guidelines)"),
            ("📘 ✗ Example", lambda: self.add_example("bad"),
             "Add the selected crops as ✗ bad examples to their class's guideline"),
        ):
            b = newButton(text, slot=slot, tip=tip)
            bar.addWidget(b)
        hint = QLabel("Double-click: open in viewer · Ctrl/Shift-click: multi-select · Ctrl+A: all")
        apply_style(hint, "color:@muted;")

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)
        lay.addWidget(hint)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.view, activated=self.delete)
        QShortcut(QKeySequence("R"), self.view, activated=self.relabel)
        self.session.samplesRemoved.connect(self._on_samples_removed)
        self.session.annotationsChanged.connect(self._on_annotations_changed)

    # ------------------------------------------------------------------ loading
    def _reset(self, store: InstanceStore):
        self.generation += 1
        self.cache.clear()
        self._inflight.clear()
        self._pending = {}
        self._changed_keys.clear()
        self.store = store
        self.model.set_order(np.zeros(0, dtype=np.int64), 0)

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None

    def show_instances(self, pairs: list[tuple[str, int]], title: str):
        """Show exactly these (file key, shape index) instances — used by Model Check drill-downs."""
        self._stop_worker()
        by_key: dict[str, set] = {}
        for k, i in pairs:
            by_key.setdefault(k, set()).add(int(i))
        store = InstanceStore()
        rank = {k: n for n, k in enumerate(by_key)}
        store.rank_of = rank.get
        errors = []
        for k, idxs in by_key.items():
            s = self.session.by_key.get(k)
            if s is None:
                continue
            try:
                store.add_rows(collect_file(self.session.adapter, s, indices=idxs))
            except Exception as exc:
                errors.append(f"{s.name}: {exc}")
        if errors:
            self.app.log_errors([f"[crop gallery] {e}" for e in errors])
        self.instance_mode = True
        self.label = title
        self._reset(store)
        self.setWindowTitle(f"{self.TITLE} — {title}")
        self._apply_sort()

    def load(self):
        label = self.label_box.currentData()
        if label is None:
            return
        self.instance_mode = False
        self.setWindowTitle(self.TITLE)
        self._stop_worker()
        split = self.split_box.currentData() or ""
        keys = set(self.app.file_dock.visible_keys()) if self.scope_box.currentIndex() == 1 else None
        idx = self.session.index if self.session.index_complete else None
        samples = []
        for s in self.session.samples:
            if split and s.split != split:
                continue
            if keys is not None and s.key not in keys:
                continue
            if idx is not None:
                c = idx.get(s.key)
                if not c or not c.get(label):
                    continue                          # the label index says: nothing to collect here
            samples.append(s)
        store = InstanceStore()
        rank = {s.key: n for n, s in enumerate(samples)}
        store.rank_of = rank.get
        self.label = label
        self._reset(store)
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, len(samples)))
        self.status.setText(f"Collecting '{label}' from {len(samples):,} file(s)…")
        self.worker = CollectWorker(self.session.adapter, samples, label)
        self.worker.progress.connect(lambda i, n: self.progress.setValue(i))
        w = self.worker                               # results of an older, stopped collection are ignored
        self.worker.batch.connect(lambda rows, w=w: self._on_batch(rows, w))
        self.worker.done.connect(lambda items, errors, w=w: self._on_collected(items, errors, w))
        self.worker.start()

    def _on_batch(self, rows, worker=None):
        if worker is not self.worker:
            return
        a, b = self.store.add_rows(rows)
        self.model.append_ids(np.arange(a, b, dtype=np.int64), len(self.store))
        self._update_status()

    def _on_collected(self, _items, errors, worker=None):
        if worker is not self.worker:
            return
        self.progress.setVisible(False)
        self.worker = None
        if errors:
            self.app.log_errors([f"[crop gallery] {e}" for e in errors])
        self._apply_sort()                            # cached crops are reused: only the order changes
        self._update_status()

    def _apply_sort(self):
        mode = self.sort_box.currentText()
        if mode == "Random":
            self._seed = shuffle_seed()
        if mode in LOOK_SORTS:
            order = self._look_order(mode)
            if order is None:
                return
        else:
            order = self.store.sorted_ids(mode, self._seed)
        self.model.set_order(order, len(self.store))
        self._update_status()

    def _look_order(self, mode):
        """Order by appearance, from the vectors of the last Visual Similarity run (instances without a vector go
        last, in file order)."""
        from AIO_QC.core.embeddings import look_order, typicality
        emb = getattr(self.app, "embed_index", None)
        ids = self.store.sorted_ids("File order")
        if not emb:
            QMessageBox.information(self, self.TITLE, "Sorting by look needs the vectors of a Visual Similarity run "
                                                      "(Shapes ▸ Visual Similarity) — run it once for these labels.")
            self.sort_box.blockSignals(True)
            self.sort_box.setCurrentIndex(1)
            self.sort_box.blockSignals(False)
            return None
        st, m = self.store, emb["map"]
        rows = np.fromiter((m.get((st.key_of(int(i)), int(st.idx[i])), -1) for i in ids), dtype=np.int64,
                           count=len(ids))
        has = rows >= 0
        if not has.any():
            self._say_status("No vectors for these crops yet — run Visual Similarity on this label first.")
            return ids
        v = emb["vecs"][rows[has]]
        sub = ids[has]
        if mode == LOOK_SORTS[0]:
            sub = sub[np.argsort(typicality(v), kind="stable")]        # least typical first
        else:
            sub = sub[look_order(v)]
        rest = ids[~has]
        if len(rest):
            self._say_status(f"{len(rest):,} crop(s) have no vector yet (added / changed after the run) — shown last.")
        return np.concatenate([sub, rest])

    def _say_status(self, text):
        self.app.statusBar().showMessage(text, 6000)

    def show_label(self, label: str, sort: str | None = None):
        """Open the gallery on a label (and a sort) — used by Visual Similarity."""
        i = self.label_box.findData(label)
        if i >= 0:
            self.label_box.setCurrentIndex(i)
        if sort:
            j = self.sort_box.findText(sort)
            if j >= 0:
                self.sort_box.blockSignals(True)
                self.sort_box.setCurrentIndex(j)
                self.sort_box.blockSignals(False)
        self.load()

    def _sel_count(self) -> int:
        sm = self.view.selectionModel()
        return sum(r.bottom() - r.top() + 1 for r in sm.selection()) if sm is not None else 0

    def _update_status(self):
        n = self.model.rowCount()
        sel = self._sel_count()
        self.status.setText(f"{n:,} crop(s) of '{self.label}' in {self.store.files:,} file(s)"
                            + (f" · {sel:,} selected" if sel else ""))

    # ---------------------------------------------------------------- thumbnails
    @property
    def tile(self) -> int:
        return self.size_slider.value()

    def _apply_tile_size(self):
        s = self.tile
        self.grid_size = QSize(s + 16, s + (30 if self.show_names else 14))
        self.view.setIconSize(QSize(s, s))
        self.view.setGridSize(self.grid_size)
        self._blank = QPixmap(s, s)
        self._blank.fill(theme_color("placeholder"))
        self._broken_pix = None

    def _resize_tiles(self):
        self._apply_tile_size()
        self._invalidate()

    def _invalidate(self):
        self.generation += 1
        self.cache.clear()
        self._inflight.clear()
        self._pending = {}
        n = self.model.rowCount()
        if n:
            self.model.dataChanged.emit(self.model.index(0), self.model.index(n - 1), [Qt.ItemDataRole.DecorationRole])

    def pixmap_for(self, iid: int) -> QPixmap:
        pix = self.cache.get(iid)
        if pix is not None:
            return pix
        if iid not in self._inflight:
            self._inflight.add(iid)
            inst = self.store.get(iid)
            color = self.app.label_colors.get(inst.label, QColor("#ff00ff"))
            # queue per image; one flush per event-loop turn batches every tile requested by this repaint
            self._pending.setdefault(inst.image, []).append((iid, inst, color))
            self._flush.start()
        return self._blank

    def _flush_pending(self):
        tile, pad = self.tile, self.pad.value() / 100
        for path, jobs in self._pending.items():
            self._priority += 1       # newest request first: what the user is looking at now
            QThreadPool.globalInstance().start(
                ImageBatchTask(self.signals, path, jobs, tile, pad, self.generation), self._priority)
        self._pending = {}

    def _on_crop(self, iid, gen, img):
        if gen != self.generation:
            return
        self._inflight.discard(iid)
        pix = QPixmap.fromImage(img) if not img.isNull() else self._broken()
        self.cache.put(iid, pix, cost=pix.width() * pix.height() * 4)
        row = self.model.row_of(iid)
        if row >= 0:
            idx = self.model.index(row)
            self.model.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])

    def _broken(self):
        if self._broken_pix is None:
            pix = QPixmap(self._blank)
            p = QPainter(pix)
            p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "no image")
            p.end()
            self._broken_pix = pix
        return self._broken_pix

    # ------------------------------------------------------------------- actions
    def selected_ids(self) -> np.ndarray:
        sm = self.view.selectionModel()
        if sm is None:
            return np.zeros(0, dtype=np.int64)
        ids = self.model.ids()
        parts = [ids[r.top():r.bottom() + 1] for r in sm.selection()]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def selected(self) -> list[Instance]:
        return [self.store.get(int(i)) for i in self.selected_ids()]

    def _open(self, index):
        iid = self.model.iid(index.row())
        self.app.open_sample(self.store.key_of(iid), int(self.store.idx[iid]))

    def _need_selection(self) -> np.ndarray:
        ids = self.selected_ids()
        if not len(ids):
            QMessageBox.information(self, self.TITLE, "Select one or more crops first.")
        return ids

    def _by_key(self, ids) -> dict[str, set]:
        out: dict[str, set] = {}
        st = self.store
        for i in ids:
            out.setdefault(st.keys[st.kid[i]], set()).add(st.idx[i])
        return out

    def relabel(self):
        ids = self._need_selection()
        if not len(ids):
            return
        st = self.store
        shown = {st.label_of(int(i)) for i in ids}
        options = list(self.session.true_classes or []) or [l for l, _ in self.session.label_totals().most_common()]
        options = [o for o in options if o not in shown] or options
        target, ok = QInputDialog.getItem(self, "Relabel", f"New label for {len(ids):,} shape(s):", options, 0,
                                          not self.session.is_yolo)
        target = target.strip()
        if not ok or not target:
            return
        if not self.session.adapter.can_use_label(target):
            QMessageBox.warning(self, self.TITLE, f"'{target}' is not valid for this dataset.")
            return
        by_key = self._by_key(ids)
        expected = {(st.key_of(int(i)), st.idx[i]): st.label_of(int(i)) for i in ids}

        def make(key, idxs):
            def fn(ann):
                hit = False
                for i in idxs:     # only if the shape still has the label we showed
                    if i < len(ann.shapes) and ann.shapes[i].get("label") == expected[(key, i)]:
                        ann.shapes[i]["label"] = target
                        hit = True
                return hit
            return fn
        changed, errors = self.session.modify_many(self, [(k, make(k, v)) for k, v in by_key.items()], "Relabelling...")
        self._refresh_keys(set(by_key))
        self._report(f"Relabelled {len(ids):,} shape(s) → '{target}' in {len(changed):,} file(s).", errors)

    def delete(self):
        ids = self._need_selection()
        if not len(ids):
            return
        by_key = self._by_key(ids)
        if QMessageBox.question(self, self.TITLE, f"Delete {len(ids):,} shape(s) from {len(by_key):,} file(s)?") \
                != QMessageBox.StandardButton.Yes:
            return
        changed, errors, removed = self.session.delete_shapes(self, by_key)
        self._refresh_keys(set(by_key))
        self._report(f"Deleted {removed:,} shape(s) in {len(changed):,} file(s).", errors)

    def add_example(self, kind: str):
        """Selected crops -> guideline examples of their class (at most 20 at once)."""
        from PyQt6.QtWidgets import QInputDialog
        insts = self.selected()[:20]
        if not insts:
            self._say_status("Select crops first")
            return
        note, ok = QInputDialog.getText(self, "Guideline example", f"Note for the {len(insts)} "
                                        f"{'good' if kind == 'good' else 'bad'} example(s) (optional):")
        if not ok:
            return
        g = self.session.guides
        n = 0
        for it in insts:
            if not it.image or not it.points:
                continue
            xs, ys = [p[0] for p in it.points], [p[1] for p in it.points]
            if g.add_example(it.label, kind, it.image, (min(xs), min(ys), max(xs), max(ys)), note,
                             src=self.session.history.rel(it.key)):
                n += 1
        self._say_status(f"📘 Added {n} {'✓ good' if kind == 'good' else '✗ bad'} example(s) to the guidelines")

    def flag_files(self):
        ids = self._need_selection()
        if len(ids):
            st = self.store
            keys = list(dict.fromkeys(st.keys[k] for k in np.unique(np.frombuffer(st.kid, dtype=np.int32)[ids])))
            self.session.set_review(keys, REVIEW_FLAG, f"crop gallery: check '{self.label}'")
            self.app.statusBar().showMessage(f"Flagged {len(keys):,} file(s)", 4000)
            self.view.viewport().update()

    def show_files(self):
        ids = self.selected_ids()
        if not len(ids):
            ids = self.model.ids()
        if len(ids):
            st = self.store
            kids = np.frombuffer(st.kid, dtype=np.int32)[ids]
            _u, first = np.unique(kids, return_index=True)
            keys = [st.keys[kids[i]] for i in np.sort(first)]          # in grid order
            self.app.apply_file_filter(keys, f"Crops of '{self.label}'")

    # ------------------------------------------------------------------- refresh after edits
    def _on_annotations_changed(self, keys):
        """Edits from anywhere (other tools, live file updates): refresh the files shown here, batched."""
        st = self.store
        hit = {k for k in keys if k in st.key_id and st.per_key[st.key_id[k]]}
        if hit:
            self._changed_keys |= hit
            self._changed_timer.start()

    def _apply_outside_changes(self):
        keys, self._changed_keys = self._changed_keys, set()
        if keys:
            self._refresh_keys(keys)

    def _refresh_keys(self, touched: set):
        """Re-collect just the touched files and put their instances where the file's crops were."""
        self._changed_keys -= touched
        st = self.store
        touched = {k for k in touched if k in st.key_id}
        if not touched:
            return
        order = self.model.ids()
        kid_arr = np.array(st.kid, dtype=np.int32)        # a copy: the store grows below (no buffer export)
        alive_arr = np.array(st.alive, dtype=np.uint8)
        tkids = np.fromiter((st.key_id[k] for k in touched), dtype=np.int32)
        mask = np.isin(kid_arr[order], tkids) if len(order) else np.zeros(0, dtype=bool)
        first_row = {}
        for r in np.nonzero(mask)[0]:
            first_row.setdefault(int(kid_arr[order[r]]), int(r))
        fresh: dict[int, list] = {}
        for k in touched:
            s = self.session.by_key.get(k)
            old_ids = np.nonzero((kid_arr == st.key_id[k]) & (alive_arr == 1))[0].tolist() if self.instance_mode else []
            rows = []
            if s is not None:
                try:
                    if self.instance_mode:
                        rows = self._reidentify(s, [st.get(i) for i in old_ids])
                    else:
                        rows = collect_file(self.session.adapter, s, self.label)
                except Exception:
                    rows = []
            fresh[st.key_id[k]] = rows
        st.kill_keys(touched)
        keep = order[~mask] if len(order) else order
        ins_at, vals = [], []
        for kid, rows in fresh.items():
            if not rows:
                continue
            a, b = st.add_rows(rows)
            r = first_row.get(kid)
            q = (r - int(mask[:r].sum())) if r is not None else len(keep)     # rows kept before the first crop
            ins_at.extend([q] * (b - a))
            vals.extend(range(a, b))
        new_order = np.insert(keep, ins_at, vals) if vals else keep
        # crops still being decoded may describe the old shape at that index -> drop every in-flight result
        self.generation += 1
        self._inflight.clear()
        self._pending = {}
        for ck in [ck for ck in list(self.cache.cache) if ck >= len(st.alive) or not st.alive[ck]]:
            self.cache.pop(ck)
        if st.needs_compaction():
            st2, remap = st.compacted()
            new_order = remap[new_order]
            new_order = new_order[new_order >= 0]
            self.cache.clear()
            self.store = st2
        self.model.set_order(new_order, len(self.store))
        self._update_status()

    def _reidentify(self, sample, old: list[Instance]) -> list[tuple]:
        """Instance mode: find each shown shape again by its outline (indices shift when shapes are deleted)."""
        if not old:
            return []
        ann = self.session.adapter.load(sample, light=True)
        from AIO_QC.core.instances import outline
        outlines = [outline(sh) for sh in ann.shapes]
        want = [[c for p in o.points for c in p] for o in old]
        found = set()
        for w in want:
            for i, o in enumerate(outlines):
                if i not in found and o is not None and len(o) == len(w) and \
                        all(abs(a - b) < 1e-3 for a, b in zip(o, w)):
                    found.add(i)
                    break
        return collect_file(self.session.adapter, sample, indices=found) if found else []

    def _report(self, text, errors):
        if errors:
            self.app.log_errors(errors)
            text += f"\n{len(errors)} error(s) — see Logs"
        self.app.statusBar().showMessage(text.replace("\n", " "), 6000)

    def _on_samples_removed(self, keys):
        st = self.store
        gone = {k for k in keys if k in st.key_id}
        if not gone:
            return
        st.kill_keys(gone)
        order = self.model.ids()
        alive = np.frombuffer(st.alive, dtype=np.uint8)
        self.model.set_order(order[alive[order] == 1] if len(order) else order, len(st))
        self._update_status()

    def closeEvent(self, event):
        self._stop_worker()
        self.generation += 1
        for sig, slot in ((self.session.samplesRemoved, self._on_samples_removed),
                          (self.session.annotationsChanged, self._on_annotations_changed)):
            try:
                sig.disconnect(slot)
            except TypeError:
                pass
        self.app.forget_dialog(self)
        super().closeEvent(event)
