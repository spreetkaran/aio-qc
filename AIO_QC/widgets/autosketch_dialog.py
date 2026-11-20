# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Auto-sketch ▸ Choose shapes: which shapes each flagged screenshot points at.

    files (left)       every file of the batch, with how many shapes are chosen; D / A or ↓ / ↑ move through them
    picture (middle)   the image with its shapes; click a shape to choose it (again to drop it); the marks the
                       screenshot will get — arrows from free space, rings or boxes — are drawn live, so what you see
                       is what is saved
    shapes (right)     the same shapes as tick boxes (#n label), All / None
    top                mark style, colour, the text at each arrow (a template, e.g. {label}) and its size — the
                       batch's options, shared with Flagged Screenshots; double-click a shape in the list to give
                       just that arrow its own text

Choices are saved per dataset (.aio_qc/autosketch.json) as you go, matched to the shapes again if a file changes.
Images are decoded on worker threads, the next ones ahead, and shown scaled: moving through files stays instant.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QSplitter, QToolButton,
    QVBoxLayout, QWidget,
)

from AIO_QC.core import autosketch as AS
from AIO_QC.utils import later, newButton, style as apply_style
from AIO_QC.core.diag import ignored as _ignored

PREVIEW_PX = 1600              # longest side of the decoded preview
CACHE = 8                      # previews kept
AHEAD = 2                      # previews decoded ahead / behind


def _decode(path, data):
    from .workers import load_qimage
    img = load_qimage(path, data)
    if img.isNull():
        return img, 0, 0
    W, H = img.width(), img.height()
    if max(W, H) > PREVIEW_PX:
        img = img.scaled(QSize(PREVIEW_PX, PREVIEW_PX), Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return img, W, H


def batch_options(settings) -> dict:
    def get(name, d, t):
        try:
            return settings.value(f"screenshot/{name}", d, type=t)
        except (TypeError, ValueError):
            return d
    from AIO_QC.core.sketch_text import TEXT_SIZES
    style = get("auto_style", "arrow_ring", str)
    text = get("auto_text", None, str)
    if text is None:                                         # before templates: a tick for "the label"
        text = "{label}" if get("auto_labels", False, bool) else ""
    size = get("auto_text_size", "M", str)
    from AIO_QC.core.sketch_text import DEFAULT_LOOK, LOOK_KEYS
    lk = get("auto_text_look", DEFAULT_LOOK, str)
    return {"text_look": lk if lk in LOOK_KEYS else DEFAULT_LOOK,"auto": get("auto", False, bool), "style": style if style in AS.STYLE_KEYS else "arrow_ring",
            "color": get("auto_color", AS.COLORS[0], str), "text": text,
            "text_size": size if size in dict(TEXT_SIZES) else "M"}


def text_scale(opts) -> float:
    from AIO_QC.core.sketch_text import TEXT_SIZES
    return dict(TEXT_SIZES).get(opts.get("text_size", "M"), 1.0)


class _Preview(QWidget):
    """The picture, every shape outlined, the chosen ones with their marks; click toggles a shape."""
    toggled = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(420, 320)
        self.img = QImage()
        self.W = self.H = 0
        self.shapes: list[dict] = []
        self.boxes: list = []
        self.chosen: set[int] = set()
        self.own: dict[int, str] = {}
        self.file = None
        self.colors: dict = {}
        self.opts = {"style": "arrow_ring", "color": AS.COLORS[0], "text": "", "text_size": "M",
                     "text_look": "auto_outline"}
        self.hover = None
        self._els: list = []
        self.message = ""

    def set_data(self, img, W, H, shapes, chosen, colors, message=""):
        self.img, self.W, self.H = img, W, H
        self.shapes = shapes
        self.boxes = [AS.shape_box(s.get("points"), s.get("shape_type")) for s in shapes]
        self.chosen = set(chosen)
        self.colors = colors
        self.message = message
        self.hover = None
        self._plan()

    def set_chosen(self, chosen, own=None):
        self.chosen = set(chosen)
        if own is not None:
            self.own = dict(own)
        self._plan()

    def set_opts(self, opts):
        self.opts = dict(opts)
        self._plan()

    def _plan(self):
        idx = sorted(i for i in self.chosen if 0 <= i < len(self.boxes) and self.boxes[i] is not None)
        others = [b for i, b in enumerate(self.boxes) if b is not None and i not in self.chosen]
        labels = AS.arrow_texts(self.shapes, idx, self.opts.get("text", ""), self.own, self.file)
        els = AS.plan(self.W, self.H, [self.boxes[i] for i in idx], others, self.opts["style"],
                      self.opts["color"], labels, text_scale=text_scale(self.opts)) if idx and self.W else []
        if els and labels:                                  # the text look, worked out on this very image
            from .sketch import restyle_texts
            els = restyle_texts(els, self.opts.get("text_look", "auto_outline"), self.opts["color"], self.img,
                                self.img.width() / max(1, self.W))
        self._els = els
        self.update()

    # ---- geometry
    def _frame(self):
        if self.img.isNull() or not self.W:
            return None
        r = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        k = min(r.width() / self.W, r.height() / self.H)
        w, h = self.W * k, self.H * k
        return QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h), k

    def _shape_at(self, pos) -> int | None:
        f = self._frame()
        if f is None:
            return None
        rect, k = f
        x, y = (pos.x() - rect.x()) / k, (pos.y() - rect.y()) / k
        tol = 6 / k
        hits = [(i, (b[2] - b[0]) * (b[3] - b[1])) for i, b in enumerate(self.boxes)
                if b is not None and b[0] - tol <= x <= b[2] + tol and b[1] - tol <= y <= b[3] + tol]
        return min(hits, key=lambda t: t[1])[0] if hits else None          # the smallest under the mouse

    # ---- events
    def mouseMoveEvent(self, e):
        h = self._shape_at(e.position())
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.CursorShape.PointingHandCursor if h is not None else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            i = self._shape_at(e.position())
            if i is not None:
                self.toggled.emit(i)

    def paintEvent(self, _e):
        from .sketch import draw_el
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(24, 25, 28))
        f = self._frame()
        if f is None:
            p.setPen(QColor(200, 200, 205))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message or "")
            return
        rect, k = f
        p.drawImage(rect, self.img)
        p.save()
        p.translate(rect.topLeft())
        p.scale(k, k)
        for i, b in enumerate(self.boxes):                  # every shape, thin: what can be chosen
            if b is None:
                continue
            chosen, hov = i in self.chosen, i == self.hover
            col = QColor(self.colors.get(str(self.shapes[i].get("label")), QColor("#34c759")))
            if not chosen:
                col.setAlphaF(0.95 if hov else 0.55)
            pen = QPen(col, (3.0 if chosen or hov else 1.4) / k)
            if not chosen and not hov:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 40) if hov else Qt.BrushStyle.NoBrush)
            s = self.shapes[i]
            st, pts = s.get("shape_type"), s.get("points") or []
            if st in ("polygon", "linestrip", "line") and len(pts) >= 2:
                path = QPainterPath(QPointF(*pts[0]))
                for q in pts[1:]:
                    path.lineTo(QPointF(*q))
                if st == "polygon":
                    path.closeSubpath()
                p.drawPath(path)
            elif st == "circle":
                p.drawEllipse(QRectF(b[0], b[1], b[2] - b[0], b[3] - b[1]))
            else:
                p.drawRect(QRectF(b[0], b[1], b[2] - b[0], b[3] - b[1]))
        for el in self._els:                                # the marks, exactly as the screenshot draws them
            draw_el(p, el)
        p.restore()
        p.end()


class AutoSketchDialog(QDialog):
    """Choose, per file, the shapes Flagged Screenshots points at (see the module docstring)."""
    TITLE = "Auto-sketch — choose shapes"
    optionsChanged = pyqtSignal(dict)

    def __init__(self, app, keys: list[str], parent=None):
        super().__init__(parent or app)
        self.app, self.session = app, app.session
        self.keys = [k for k in keys if k in self.session.by_key]
        self.store = getattr(self.session, "_autosketch", None)
        if self.store is None:
            self.store = self.session._autosketch = AS.ChoiceStore(self.session.root)
        self.opts = batch_options(app.settings)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="autosketch")
        self._cache: OrderedDict = OrderedDict()            # key -> Future (img, W, H)
        self._ann: dict = {}                                # key -> shapes in image px
        self.cur = None
        self.setWindowTitle(self.TITLE)
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        lay = QVBoxLayout(self)
        # ---- options
        top = QHBoxLayout()
        top.addWidget(QLabel("Mark:"))
        self.style_box = QComboBox()
        for k, text in AS.STYLES:
            self.style_box.addItem(text, k)
        self.style_box.setCurrentIndex(max(0, self.style_box.findData(self.opts["style"])))
        self.style_box.currentIndexChanged.connect(self._opts_changed)
        top.addWidget(self.style_box)
        top.addSpacing(10)
        top.addWidget(QLabel("Colour:"))
        self.swatches = []
        for c in AS.COLORS:
            b = QToolButton()
            b.setCheckable(True)
            b.setFixedSize(22, 22)
            b.setToolTip(c)
            b.setStyleSheet(f"QToolButton{{background:{c}; border:1px solid #888; border-radius:11px;}}"
                            "QToolButton:checked{border:3px solid palette(highlight);}")
            b.clicked.connect(lambda _c=False, col=c: self._set_color(col))
            self.swatches.append((c, b))
            top.addWidget(b)
        top.addSpacing(10)
        from AIO_QC.core.sketch_text import TEXT_SIZES
        top.addWidget(QLabel("Text:"))
        self.text_box = QComboBox()
        self.text_box.setEditable(True)
        self.text_box.setMinimumWidth(190)
        self.text_box.setToolTip("Written at every arrow — pick a template or type; {label} the shape's label, {n} "
                                 "its number, {file} the file. Double-click a shape in the list for its own text.")
        self.text_box.lineEdit().setPlaceholderText("no text")
        self._fill_templates(self.opts["text"])
        self.text_box.currentTextChanged.connect(self._opts_changed)
        top.addWidget(self.text_box)
        self.size_box = QComboBox()
        for k, _v in TEXT_SIZES:
            self.size_box.addItem(k, k)
        self.size_box.setToolTip("Text size: S · M · L · XL (M follows the arrow's size, which follows the image)")
        self.size_box.setCurrentIndex(max(0, self.size_box.findData(self.opts["text_size"])))
        self.size_box.currentIndexChanged.connect(self._opts_changed)
        top.addWidget(self.size_box)
        from AIO_QC.core.sketch_text import LOOKS
        self.look_box = QComboBox()
        for k, name, *_r in LOOKS:
            self.look_box.addItem(name, k)
        self.look_box.setToolTip("How the text looks — style and colours; “for the background”: chosen per text to "
                                 "stand out from the part of the image it is put on")
        self.look_box.setCurrentIndex(max(0, self.look_box.findData(self.opts["text_look"])))
        self.look_box.currentIndexChanged.connect(self._opts_changed)
        top.addWidget(self.look_box)
        top.addWidget(newButton("Templates…", slot=self._edit_templates, tip="Add, edit or re-order the text templates"))
        top.addStretch(1)
        self.summary = QLabel()
        apply_style(self.summary, "color:@muted;")
        top.addWidget(self.summary)
        lay.addLayout(top)
        # ---- files | picture | shapes
        split = QSplitter(Qt.Orientation.Horizontal)
        self.files = QListWidget()
        self.files.currentRowChanged.connect(self._show_row)
        split.addWidget(self.files)
        self.preview = _Preview()
        self.preview.toggled.connect(self._toggle)
        split.addWidget(self.preview)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        self.shapes = QListWidget()
        self.shapes.itemChanged.connect(self._shape_ticked)
        self.shapes.itemDoubleClicked.connect(self._own_text)
        sl.addWidget(self.shapes, 1)
        row = QHBoxLayout()
        row.addWidget(newButton("All", slot=lambda: self._set_all(True), tip="Choose every shape of this image"))
        row.addWidget(newButton("None", slot=lambda: self._set_all(False), tip="Choose no shape of this image"))
        sl.addLayout(row)
        split.addWidget(side)
        split.setSizes([240, 800, 240])
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        foot = QHBoxLayout()
        hint = QLabel("Click shapes on the picture (or tick them) — the marks show as they will be saved · "
                      "double-click a shape in the list: its own text · D / A next / previous image · files "
                      "without chosen shapes are saved as usual")
        hint.setWordWrap(True)
        apply_style(hint, "color:@muted;")
        foot.addWidget(hint, 1)
        done = newButton("Done", slot=self.accept)
        done.setDefault(True)
        foot.addWidget(done)
        lay.addLayout(foot)
        for seq, step in (("D", 1), ("Down", 1), ("A", -1), ("Up", -1)):
            QShortcut(QKeySequence(seq), self, activated=lambda s=step: self._step(s))
        self._save_t = QTimer(self, singleShot=True, interval=600, timeout=self.store.flush)
        self._sync_swatches()
        self.preview.set_opts(self.opts)
        self._fill_files()
        if self.keys:
            self.files.setCurrentRow(0)
        else:
            self.preview.message = "No files in this batch"

    # ---- files
    def _file_text(self, key):
        n = self.store.count(key)
        return f"{Path(key).name}" + (f"  ·  {n} chosen" if n else "")

    def _fill_files(self):
        self.files.blockSignals(True)
        self.files.clear()
        for k in self.keys:
            it = QListWidgetItem(self._file_text(k))
            it.setToolTip(k)
            self.files.addItem(it)
        self.files.blockSignals(False)
        self._update_summary()

    def _update_summary(self):
        n = sum(1 for k in self.keys if self.store.count(k))
        self.summary.setText(f"{n:,} of {len(self.keys):,} file(s) with chosen shapes")

    def _step(self, d):
        r = self.files.currentRow() + d
        if 0 <= r < self.files.count():
            self.files.setCurrentRow(r)

    def _shapes_of(self, key):
        if key not in self._ann:
            s = self.session.by_key[key]
            try:
                ann = self.session.adapter.load(s)
                self._ann[key] = (ann, None)
            except Exception as exc:
                _ignored(exc)
                self._ann[key] = (None, str(exc))
        return self._ann[key]

    def _prefetch(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ann, _err = self._shapes_of(key)
        s = self.session.by_key[key]
        path = (ann.image_path if ann is not None and ann.image_path else None) or s.image_path
        fut = self._pool.submit(_decode, path, ann.image_data if ann is not None else None)
        self._cache[key] = fut
        while len(self._cache) > CACHE:
            self._cache.popitem(last=False)
        return fut

    def _show_row(self, r):
        if not 0 <= r < len(self.keys):
            return
        key = self.keys[r]
        self.cur = key
        fut = self._prefetch(key)
        for d in range(1, AHEAD + 1):                      # the neighbours decode meanwhile
            for j in (r + d, r - d):
                if 0 <= j < len(self.keys):
                    self._prefetch(self.keys[j])
        if fut.done():
            self._apply(key, fut)
        else:
            self.preview.set_data(QImage(), 0, 0, [], [], {}, "Loading …")
            self._wait(key, fut)

    def _wait(self, key, fut):
        if self.cur != key:
            return
        if fut.done():
            self._apply(key, fut)
        else:
            later(15, self, lambda: self._wait(key, fut))

    def _apply(self, key, fut):
        if self.cur != key:
            return
        try:
            img, W, H = fut.result()
        except Exception as exc:
            _ignored(exc)
            img, W, H = QImage(), 0, 0
            self.preview.message = f"Could not read the image: {exc}"
        ann, err = self._shapes_of(key)
        shapes = []
        if ann is not None and W:
            shapes = list((ann.scaled(W, H) if getattr(ann, "normalized", False) else ann).shapes)
        self._cur_shapes = shapes
        chosen, own = self.store.resolve(key, shapes, with_texts=True)
        self._own = dict(own)
        msg = "" if not img.isNull() else (err or "Could not read the image")
        self.preview.own, self.preview.file = dict(own), Path(key).name
        self.preview.set_data(img, W, H, shapes, chosen, self.app.label_colors, msg)
        self._fill_shapes(shapes, chosen)

    # ---- shapes
    def _fill_shapes(self, shapes, chosen):
        self.shapes.blockSignals(True)
        self.shapes.clear()
        for i, s in enumerate(shapes):
            own = getattr(self, "_own", {}).get(i)
            it = QListWidgetItem(f"#{i + 1}  {s.get('label')}" + (f"  — “{own}”" if own else ""))
            it.setToolTip("Double-click: this arrow's own text")
            it.setData(Qt.ItemDataRole.UserRole, i)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if i in chosen else Qt.CheckState.Unchecked)
            col = self.app.label_colors.get(str(s.get("label")))
            if col is not None:
                it.setForeground(QColor(col))
            self.shapes.addItem(it)
        if not shapes:
            it = QListWidgetItem("(no shapes)")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.shapes.addItem(it)
        self.shapes.blockSignals(False)

    def _chosen_now(self):
        return {self.shapes.item(r).data(Qt.ItemDataRole.UserRole) for r in range(self.shapes.count())
                if self.shapes.item(r).checkState() == Qt.CheckState.Checked}

    def _commit(self, chosen):
        key = self.cur
        if key is None:
            return
        own = {i: s for i, s in getattr(self, "_own", {}).items() if i in chosen}
        self.store.set(key, getattr(self, "_cur_shapes", []), chosen, own)
        self._save_t.start()
        self.preview.set_chosen(chosen, own)
        r = self.keys.index(key)
        self.files.item(r).setText(self._file_text(key))
        self._update_summary()

    def _shape_ticked(self, _it):
        self._commit(self._chosen_now())

    def _toggle(self, i):
        chosen = self._chosen_now() ^ {i}
        self.shapes.blockSignals(True)
        for r in range(self.shapes.count()):
            it = self.shapes.item(r)
            if it.data(Qt.ItemDataRole.UserRole) == i:
                it.setCheckState(Qt.CheckState.Checked if i in chosen else Qt.CheckState.Unchecked)
                self.shapes.scrollToItem(it)
        self.shapes.blockSignals(False)
        self._commit(chosen)

    def _set_all(self, on):
        n = len(getattr(self, "_cur_shapes", []))
        self.shapes.blockSignals(True)
        for r in range(self.shapes.count()):
            it = self.shapes.item(r)
            if it.data(Qt.ItemDataRole.UserRole) is not None:
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.shapes.blockSignals(False)
        self._commit(set(range(n)) if on else set())

    def _own_text(self, it):
        """Double-click a shape: the text at just its arrow (empty: the batch's text). Chooses the shape too."""
        from AIO_QC.core.sketch_text import expand
        from .sketch_text_dialog import TextDialog
        i = it.data(Qt.ItemDataRole.UserRole)
        if i is None:
            return
        shapes = getattr(self, "_cur_shapes", [])
        s = shapes[i] if 0 <= i < len(shapes) else {}
        cur = self._own.get(i) or expand(self.opts.get("text", ""), label=str(s.get("label")), n=i + 1,
                                         file=Path(self.cur).name if self.cur else None)
        spot, fs = self._text_spot(i)
        p = self.preview
        dlg = TextDialog(self.app.settings, self, text=cur, title=f"Text at the arrow — #{i + 1} {s.get('label')}",
                         fields={"label": str(s.get("label")), "n": i + 1,
                                 "file": Path(self.cur).name if self.cur else None}, allow_empty=True,
                         image=p.img, scale=p.img.width() / max(1, p.W), point=spot, sketch_color=self.opts["color"],
                         preview_size=fs, fixed_look=self.opts.get("text_look"))   # the batch's look
        if not dlg.exec():
            return
        txt = dlg.text()
        batch = expand(self.opts.get("text", ""), label=str(s.get("label")), n=i + 1,
                       file=Path(self.cur).name if self.cur else None)
        if txt and txt != batch:
            self._own[i] = txt
        else:
            self._own.pop(i, None)                     # the same as the batch's text: nothing of its own
        chosen = self._chosen_now() | {i}
        self._fill_shapes(shapes, chosen)
        self._commit(chosen)

    def _text_spot(self, i):
        """Where the text of shape i's arrow goes and its size (from the planned marks), for the preview."""
        p = self.preview
        b = p.boxes[i] if 0 <= i < len(p.boxes) else None
        idx = sorted(j for j in p.chosen | {i} if 0 <= j < len(p.boxes) and p.boxes[j] is not None)
        others = [bb for j, bb in enumerate(p.boxes) if bb is not None and j not in idx]
        els = AS.plan(p.W, p.H, [p.boxes[j] for j in idx], others, self.opts["style"], self.opts["color"],
                      ["x"] * len(idx), text_scale=text_scale(self.opts)) if p.W else []
        texts = [e for e in els if e["t"] == "text"]
        k = idx.index(i) if i in idx else -1
        if 0 <= k < len(texts):
            return tuple(texts[k]["pts"][0]), texts[k].get("fs", 40)
        fs = AS.unit(p.W or 1, p.H or 1) * 25 * text_scale(self.opts)
        return ((b[0], max(fs * 1.2, b[1] - fs * 0.4)) if b else (fs, fs * 1.5)), fs

    def _fill_templates(self, current):
        from AIO_QC.core.sketch_text import load_templates
        self.text_box.blockSignals(True)
        self.text_box.clear()
        self.text_box.addItem("")                       # no text
        for s in load_templates(self.app.settings):
            self.text_box.addItem(s)
        self.text_box.setEditText(current or "")
        self.text_box.blockSignals(False)

    def _edit_templates(self):
        from .sketch_text_dialog import TemplateEditor
        if TemplateEditor(self.app.settings, self).exec():
            self._fill_templates(self.text_box.currentText())

    # ---- options
    def _set_color(self, c):
        self.opts["color"] = c
        self._sync_swatches()
        self._opts_changed()

    def _sync_swatches(self):
        for c, b in self.swatches:
            b.setChecked(c.lower() == str(self.opts["color"]).lower())

    def _opts_changed(self, *_a):
        self.opts["style"] = self.style_box.currentData()
        self.opts["text"] = self.text_box.currentText().strip()
        self.opts["text_size"] = self.size_box.currentData()
        self.opts["text_look"] = self.look_box.currentData()
        st = self.app.settings
        st.setValue("screenshot/auto_style", self.opts["style"])
        st.setValue("screenshot/auto_color", self.opts["color"])
        st.setValue("screenshot/auto_text", self.opts["text"])
        st.setValue("screenshot/auto_text_size", self.opts["text_size"])
        st.setValue("screenshot/auto_text_look", self.opts["text_look"])
        self.preview.set_opts(self.opts)
        self.optionsChanged.emit(dict(self.opts))

    def done(self, r):
        self._save_t.stop()
        self.store.flush()
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().done(r)
