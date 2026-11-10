# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Screenshots of many files at once (Review ▸ Flagged Screenshots, File list ▸ Save screenshots of N files).

Each file is drawn on a hidden copy of the canvas with the window's current look — ticked labels, outlines /
fill / tags, brightness / contrast (a file's own adjustment from this session first), the model predictions
layer — so a batch screenshot looks exactly like Image QC. The view on screen is never touched.

    view (Settings ▸ Screenshots ▸ Batch view)
      spot   the same spot as now: your current zoom and position, carried to every file like Keep Previous Zoom
             (same-size files: the same pixel; other sizes: the same part of the image); fit stays fit
      whole  the whole image at full resolution; outlines and tags scaled up so they read as on a ~1280 px view
    caption  a strip under the picture: file, review mark and time, the flag note
    sketch   the file's saved sketch drawing (Sketch on Image)

Pipeline: images are decoded on worker threads a few files ahead, each file is drawn on the GUI thread (one per
event-loop turn, so the app stays responsive) and written on a worker thread. Progress and Cancel are in the
status bar's task indicator; a notification sums up at the end.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QObject, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPixmap, QTextOption
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QRadioButton,
    QVBoxLayout, QWidget,
)

from AIO_QC.core.state import REVIEW_FLAG

from .screenshot import capture_view, save_image, unique_path

VIEWS = [("spot", "Same spot as now — the current zoom and position on every file"),
         ("whole", "Whole image — full resolution")]
EXISTING = [("keep", "Keep both (name_2.png)"), ("overwrite", "Overwrite"), ("skip", "Skip")]
DEFAULTS = {"batch_view": "spot", "caption": True, "sketch": True, "existing": "keep"}
REF_PX = 1280                    # whole image: outlines / tags look as they do on a view this many pixels across
AHEAD = 3                        # images decoded ahead of the one being drawn
WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))   # decode + encode; one core stays free for the window
REVIEW_TEXT = {"ok": "✓ OK", "flag": "⚑ Flagged", "fixed": "🔧 Fixed"}


def option(settings, name):
    d = DEFAULTS[name]
    try:
        return settings.value(f"screenshot/{name}", d, type=bool if isinstance(d, bool) else str)
    except (TypeError, ValueError):
        return d


# ====================================================================================================== drawing
class ShotRenderer:
    """Draws one file the way the window shows it, on a canvas that is never on screen."""

    def __init__(self, app, view: str, viewport_size=None):
        from .canvas import Canvas
        self.app, self.view = app, view
        main = app.canvas
        c = self.c = Canvas()
        c.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        c.editable = False
        size = viewport_size or main.size()
        c.resize(size)
        c.show()
        self.state = main.view_state() if view == "spot" and not main._original.isNull() else None
        self.style = (main.show_tags, main.fill_shapes, main.fill_alpha, main.outline_scale,
                      getattr(main, "tag_scale", 1.0), main.pixmap_item.pixel_view)
        self.display = dict(main.display)
        self._sketch_store = None

    def close(self):
        self.c.close()
        self.c.deleteLater()

    def _display_for(self, key) -> dict:
        own = getattr(self.app, "_display_by_key", {}).get(key)
        if own is None:
            return dict(self.display)
        return {"brightness": own["brightness"], "contrast": own["contrast"], "gamma": own["gamma"] / 100,
                "auto": bool(own["auto"])}

    def _sketch(self, key) -> list:
        from .sketch import SketchStore
        s = self.app.session
        if self._sketch_store is None:
            self._sketch_store = getattr(s, "_sketches", None) or SketchStore(s.root)
        return self._sketch_store.load(key)

    def render(self, key: str, img: QImage, ann, with_sketch: bool) -> QImage:
        c, app = self.c, self.app
        tags, fill, alpha, outline, tag_scale, pixel_view = self.style
        W, H = img.width(), img.height()
        dpr = c.viewport().devicePixelRatioF() or 1.0
        k = 1.0
        if self.view == "whole":
            c.resize(max(1, math.ceil(W / dpr)), max(1, math.ceil(H / dpr)))     # 1 image px = 1 device px
            k = max(1.0, max(W, H) / dpr / REF_PX)
        c.show_tags, c.fill_shapes, c.fill_alpha = tags, fill, alpha
        c.outline_scale, c.tag_scale = outline * k, tag_scale * k
        c.pixmap_item.pixel_view = pixel_view
        c.display = self._display_for(key)
        c.keep_zoom = False
        c.loadPixmap(QPixmap.fromImage(img))
        if self.view == "whole":
            c._apply_view(1.0 / dpr, c._image_center())
            c._finish_zoom(c.CUSTOM_MODE)
        elif self.state is not None:
            c.apply_view_state(self.state)
        shapes = list(ann.shapes) if ann is not None else []
        if ann is not None and getattr(ann, "normalized", False):
            shapes = ann.scaled(W, H).shapes
        c.loadShapes(shapes, app.label_colors, set(app.visible_labels))
        items = app.pred_layer_items(key, W) if hasattr(app, "pred_layer_items") else []
        if items:
            c.set_pred_layer(items)
        layer = None
        if with_sketch:
            els = self._sketch(key)
            if els:
                from .sketch import SketchLayer
                layer = SketchLayer(W, H)
                layer.render_all(els)
                c.scene.addItem(layer)
        try:
            c.layout_tags() if c.shape_groups else None
            return capture_view(c)
        finally:
            if layer is not None:
                c.scene.removeItem(layer)


def caption_strip(img: QImage, title: str, meta: str, note: str) -> QImage:
    """The picture with a dark strip under it: file (bold) · review mark · time, then the note (wrapped).
    Pure QImage painting: safe on a worker thread."""
    w = img.width()
    px = max(12, min(40, round(w / 70)))
    pad = round(px * 0.8)
    f1, f2 = QFont(), QFont()
    f1.setPixelSize(px)
    f1.setBold(True)
    f2.setPixelSize(round(px * 0.92))
    fm2 = QFontMetrics(f2)
    line1 = QFontMetrics(f1).height()
    inner = w - 2 * pad
    note_rect = fm2.boundingRect(0, 0, max(1, inner), 100000, int(Qt.TextFlag.TextWordWrap), note) if note else None
    h = pad + line1 + (round(px * 0.35) + note_rect.height() if note_rect else 0) + pad
    out = QImage(w, img.height() + h, QImage.Format.Format_RGB32)
    p = QPainter(out)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)      # a plain copy, no blending
    p.drawImage(0, 0, img)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    p.fillRect(0, img.height(), w, h, QColor(28, 28, 30))                     # only the strip is filled
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    y = img.height() + pad
    p.setFont(f1)
    p.setPen(QColor(245, 245, 245))
    fm1 = QFontMetrics(f1)
    title_w = min(fm1.horizontalAdvance(title), inner)
    p.drawText(QRectF(pad, y, title_w, line1), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               fm1.elidedText(title, Qt.TextElideMode.ElideMiddle, inner))
    if meta:
        p.setFont(f2)
        p.setPen(QColor(190, 190, 195))
        mx = pad + title_w + px
        if mx < w - pad:
            p.drawText(QRectF(mx, y, w - pad - mx, line1), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       fm2.elidedText(meta, Qt.TextElideMode.ElideRight, int(w - pad - mx)))
    if note_rect:
        p.setFont(f2)
        p.setPen(QColor(255, 214, 102))
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WordWrap)
        p.drawText(QRectF(pad, y + line1 + round(px * 0.35), inner, note_rect.height()), note, opt)
    p.end()
    return out


def _finish_file(pic: QImage, caption, target: Path, fmt: str) -> bool:
    """Worker thread: the caption strip and the encoding (the GUI thread only draws the view)."""
    if caption is not None:
        pic = caption_strip(pic, *caption)
    return save_image(pic, target, fmt)


# ====================================================================================================== the job
class BatchShots(QObject):
    """Screenshots of `keys`, in the background (see the module docstring)."""

    def __init__(self, app, keys: list[str], view: str, caption: bool, sketch: bool, existing: str,
                 folder: Path | None = None):
        super().__init__(app)
        self.app, self.keys = app, list(keys)
        self.view, self.caption, self.sketch, self.existing = view, caption, sketch, existing
        self.folder = folder
        self.done = self.written = self.skipped = 0
        self.errors: list[str] = []
        self.paths: list[Path] = []
        self.cancelled = False
        self.finished = False
        self._i = 0
        self._pending: dict[int, object] = {}        # index -> Future (decoded image)
        self._saves: list = []
        self._pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="shots")
        self._renderer: ShotRenderer | None = None
        self._task = None
        self._t0 = time.monotonic()
        self.on_finished = None                      # callable(self) (tests, dialog)

    # ---- life cycle
    def start(self):
        from AIO_QC.widgets import tasks
        self._renderer = ShotRenderer(self.app, self.view)
        self._task = tasks.start(f"Screenshots · {len(self.keys):,} files", cancel=self.cancel)
        self._fill()
        QTimer.singleShot(0, self._step)
        return self

    def cancel(self):
        self.cancelled = True

    def _target(self, key) -> Path:
        shots = self.app.shots
        t = shots.target_for(key)
        if self.folder is not None:
            t = self.folder / t.relative_to(shots.folder)
        return t

    # ---- decode ahead (worker threads)
    def _fill(self):
        s = self.app.session
        while len(self._pending) < AHEAD and self._i + len(self._pending) < len(self.keys):
            j = self._i + len(self._pending)
            key = self.keys[j]
            sample = s.by_key.get(key) if s else None
            if sample is None:
                self._pending[j] = None
                continue
            try:
                ann = s.adapter.load(sample)
            except Exception as exc:                 # unreadable annotation: the picture is still useful
                ann, why = None, str(exc)
                logger.debug(f"[screenshots] {sample.name}: {why}")
            from .workers import load_qimage
            path = (ann.image_path if ann is not None and ann.image_path else None) or sample.image_path
            data = ann.image_data if ann is not None else None
            self._pending[j] = (ann, self._pool.submit(load_qimage, path, data))

    def _step(self):
        if self.finished:
            return
        if self.cancelled or self._i >= len(self.keys):
            self._finish()
            return
        item = self._pending.get(self._i)
        if item is not None and not item[1].done():
            QTimer.singleShot(5, self._step)         # the decode of this file is still running
            return
        key = self.keys[self._i]
        self._pending.pop(self._i, None)
        self._i += 1
        try:
            self._one(key, item)
        except Exception as exc:
            logger.exception("screenshot failed")
            self.errors.append(f"{Path(key).name}: {exc}")
        self.done += 1
        if self._task is not None:
            self._task.update(self.done / max(1, len(self.keys)), f"{self.done:,} / {len(self.keys):,}")
        self._fill()
        QTimer.singleShot(0, self._step)

    def _one(self, key, item):
        name = Path(key).name
        if item is None:
            self.errors.append(f"{name}: not in the dataset any more")
            return
        target = self._target(key)
        if self.existing == "skip" and target.exists():
            self.skipped += 1
            return
        ann, fut = item
        img = fut.result()
        if img is None or img.isNull():
            self.errors.append(f"{name}: the image could not be read")
            return
        pic = self._renderer.render(key, img, ann, self.sketch)
        if pic.isNull():
            self.errors.append(f"{name}: nothing to draw")
            return
        caption = self._caption_text(key) if self.caption else None
        if self.existing == "keep":
            target = unique_path(target)
            while target in self.paths:              # two files in this batch: never the same name
                target = unique_path(target.with_name(target.stem + "_b" + target.suffix))
        fmt = self.app.shots.fmt
        self.paths.append(target)
        self._saves.append((name, target, self._pool.submit(_finish_file, pic, caption, target, fmt)))

    def _caption_text(self, key):
        s = self.app.session
        try:
            title = Path(key).relative_to(s.root).as_posix()
        except ValueError:
            title = Path(key).name
        rv = s.review.get(key) if s and s.review else None
        meta = ""
        if rv:
            meta = REVIEW_TEXT.get(rv.get("status"), rv.get("status") or "")
            if rv.get("ts"):
                meta += f" · {rv['ts']}"
        note = (rv or {}).get("note") or ""
        return title, meta, note

    def _finish(self):
        if self.finished:
            return
        if any(not f.done() for _n, _t, f in self._saves):
            QTimer.singleShot(20, self._finish)      # the last files are still being written
            return
        self.finished = True
        for name, target, f in self._saves:
            try:
                ok = f.result()
            except Exception as exc:
                ok, why = False, str(exc)
            else:
                why = "could not be written"
            if ok:
                self.written += 1
            else:
                self.errors.append(f"{name}: {target} {why}")
        for _j, item in self._pending.items():
            if item is not None:
                item[1].cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._task is not None:
            self._task.finish()
        self._report()
        if callable(self.on_finished):
            self.on_finished(self)

    def _report(self):
        app = self.app
        secs = time.monotonic() - self._t0
        parts = [f"{self.written:,} saved"]
        if self.skipped:
            parts.append(f"{self.skipped:,} skipped (already there)")
        if self.errors:
            parts.append(f"{len(self.errors):,} failed")
        head = "Screenshots cancelled" if self.cancelled else "Screenshots ready"
        folder = self.paths[0].parent if self.paths else self._target(self.keys[0]).parent if self.keys else None
        if folder is not None:
            app.shots.last = self.paths[0] if self.paths else None
        logger.info(f"[screenshots] {' · '.join(parts)} in {secs:.1f}s")
        toast = getattr(app, "toast", None)
        if toast is not None:
            details = (lambda e=list(self.errors): "\n".join(e)) if self.errors else None
            toast.show_message(f"<b>{head}</b><br>{' · '.join(parts)} — {secs:.0f} s",
                               kind="warn" if self.errors else "success", key="batch_shots",
                               actions=[("Open folder", app.shots.open_folder)], details=details,
                               timeout_ms=10000)
        app.statusBar().showMessage(f"{head}: {' · '.join(parts)}", 8000)


# ====================================================================================================== the dialog
class BatchShotsDialog(QDialog):
    """Which files and how; the work itself runs in the background once started."""
    SUPPORTED = None
    TITLE = "Flagged Screenshots"

    def __init__(self, app, keys: list[str] | None = None):
        super().__init__(app)
        self.app = app
        self.st = app.settings
        self._explicit = list(keys) if keys else None
        self.job: BatchShots | None = None
        self.setWindowTitle(self.TITLE)
        lay = QVBoxLayout(self)
        intro = QLabel("Save a screenshot of every file — each one drawn exactly as Image QC shows it (labels you "
                       "ticked, outlines, brightness / contrast, predictions layer), in the background.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        self.scope = QComboBox()
        self.scope.currentIndexChanged.connect(lambda _i: self._update())
        form.addRow("Files:", self.scope)
        vw = QWidget()
        vl = QVBoxLayout(vw)
        vl.setContentsMargins(0, 0, 0, 0)
        self.view_group = QButtonGroup(self)
        cur = option(self.st, "batch_view")
        self.view_btns = {}
        for i, (k, text) in enumerate(VIEWS):
            rb = QRadioButton(text)
            rb.setChecked(k == cur)
            self.view_group.addButton(rb, i)
            self.view_btns[k] = rb
            vl.addWidget(rb)
        form.addRow("View:", vw)
        self.caption = QCheckBox("Caption strip — file, review mark, flag note")
        self.caption.setChecked(option(self.st, "caption"))
        self.sketch = QCheckBox("Include sketch drawings")
        self.sketch.setChecked(option(self.st, "sketch"))
        form.addRow("", self.caption)
        form.addRow("", self.sketch)
        self.existing = QComboBox()
        for k, text in EXISTING:
            self.existing.addItem(text, k)
        self.existing.setCurrentIndex(max(0, [k for k, _t in EXISTING].index(option(self.st, "existing"))
                                          if option(self.st, "existing") in dict(EXISTING) else 0))
        form.addRow("Existing files:", self.existing)
        fr = QWidget()
        fl = QHBoxLayout(fr)
        fl.setContentsMargins(0, 0, 0, 0)
        self.folder_lbl = QLabel()
        self.folder_lbl.setWordWrap(True)
        self.folder_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fl.addWidget(self.folder_lbl, 1)
        from AIO_QC.utils import newButton
        fl.addWidget(newButton("Change…", icon="open", slot=self._change_folder,
                               tip="The screenshots folder (Settings ▸ Screenshots)"))
        form.addRow("Folder:", fr)
        lay.addLayout(form)
        self.note = QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        lay.addStretch(1)
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.start_btn = newButton("Save screenshots", icon="screenshot", slot=self.start)
        self.start_btn.setDefault(True)
        foot.addWidget(self.start_btn)
        foot.addWidget(newButton("Close", slot=self.close))
        lay.addLayout(foot)
        self.resize(560, 360)
        self.refresh()

    def set_keys(self, keys):
        self._explicit = list(keys) if keys else None
        self.refresh()

    def _scope_options(self):
        s, dock = self.app.session, self.app.file_dock
        if s is None:
            return []
        flagged = s.keys_with_review({REVIEW_FLAG}, rank=dock.model.row_of)
        opts = []
        if self._explicit:
            opts.append((f"Selected files ({len(self._explicit):,})", list(self._explicit)))
        opts.append((f"All flagged files ({len(flagged):,})", flagged))
        if dock.proxy.filtering():
            shown = dock.visible_keys()
            seen = set(shown)
            opts.append((f"Flagged files in the current list ({sum(k in seen for k in flagged):,})",
                         [k for k in flagged if k in seen]))
            opts.append((f"All files in the current list ({len(shown):,})", shown))
        return opts

    def refresh(self):
        cur = self.scope.currentText()
        self.scope.blockSignals(True)
        self.scope.clear()
        for text, keys in self._scope_options():
            self.scope.addItem(text, keys)
        i = self.scope.findText(cur)
        self.scope.setCurrentIndex(i if i >= 0 and not self._explicit else 0)
        self.scope.blockSignals(False)
        self.folder_lbl.setText(str(self.app.shots.folder / (self.app.session.root.name if self.app.session else "")))
        self._update()

    def _update(self):
        keys = self.scope.currentData() or []
        running = self.job is not None and not self.job.finished
        self.start_btn.setEnabled(bool(keys) and not running)
        if running:
            self.note.setText("Running — see the status bar (Cancel there).")
        elif not keys:
            self.note.setText("No files in this choice — flag files first (X), or select files in the file list.")
        else:
            self.note.setText(f"{len(keys):,} file(s) · {self.app.shots.fmt.upper()} · the same folder layout as E.")

    def _change_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Screenshots folder", str(self.app.shots.folder))
        if d:
            self.st.setValue("screenshot/dir", d)
            self.refresh()

    def options(self) -> dict:
        view = next(k for k, rb in self.view_btns.items() if rb.isChecked())
        return {"batch_view": view, "caption": self.caption.isChecked(), "sketch": self.sketch.isChecked(),
                "existing": self.existing.currentData()}

    def start(self) -> BatchShots | None:
        keys = self.scope.currentData() or []
        if not keys or (self.job is not None and not self.job.finished):
            return None
        o = self.options()
        for k, v in o.items():                       # the dialog's choices become the defaults (Settings)
            self.st.setValue(f"screenshot/{k}", v)
        self.job = BatchShots(self.app, keys, o["batch_view"], o["caption"], o["sketch"], o["existing"])
        self.job.on_finished = lambda _j: self._update()
        self.job.start()
        self._update()
        self.close()
        return self.job

