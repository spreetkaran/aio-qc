# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The text window of Sketch on Image's Text tool (and auto-sketch's per-shape text): templates to click, the text,
its size, its look (style + colours, "auto" ones chosen for the background) and a live preview of the text on the very
part of the image it will be put on — and the template list editor shared with auto-sketch.
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
    QSpinBox, QVBoxLayout, QWidget,
)

from AIO_QC.core import sketch_text as ST
from AIO_QC.utils import newButton, style as apply_style


class TextPreview(QWidget):
    """The text as it will look, on the part of the image it lands on (or a neutral checkerboard without one)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.el = None
        self.img, self.scale = None, 1.0

    def set_background(self, img: QImage | None, scale: float = 1.0):
        self.img, self.scale = img, scale
        self.update()

    def set_el(self, el):
        self.el = el
        self.update()

    def paintEvent(self, _e):
        from .sketch import draw_el, text_rect
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(40, 41, 45))
        if not self.el or not self.el.get("text"):
            p.setPen(QColor(170, 170, 175))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Preview")
            return
        tr = text_rect(self.el)
        m = max(tr.height() * 0.6, 12)
        area = tr.adjusted(-m, -m, m, m)                     # the text and some of what is around it
        k = min((self.width() - 12) / max(area.width(), 1), (self.height() - 12) / max(area.height(), 1), 4.0)
        w, h = area.width() * k, area.height() * k
        dst = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.save()
        p.setClipRect(dst)
        if self.img is not None and not self.img.isNull():
            s = self.scale
            p.drawImage(dst, self.img, QRectF(area.x() * s, area.y() * s, area.width() * s, area.height() * s))
        else:                                                # no image: a checkerboard of light and dark
            c = max(8, int(dst.height() / 6))
            for yy in range(int(dst.top()), int(dst.bottom()), c):
                for xx in range(int(dst.left()), int(dst.right()), c):
                    light = ((xx - int(dst.left())) // c + (yy - int(dst.top())) // c) % 2
                    p.fillRect(xx, yy, c, c, QColor(225, 225, 228) if light else QColor(70, 70, 75))
        p.translate(dst.topLeft())
        p.scale(k, k)
        p.translate(-area.topLeft())
        draw_el(p, self.el)
        p.restore()
        p.setPen(QColor(120, 120, 128))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(dst.adjusted(-0.5, -0.5, 0.5, 0.5))
        p.end()


class TemplateEditor(QDialog):
    """Add, edit (double-click), remove and re-order the text templates; Reset brings the built-in list back."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.st = settings
        self.setWindowTitle("Text templates")
        self.resize(460, 420)
        v = QVBoxLayout(self)
        info = QLabel("Fields: {label} the shape's label · {n} its number · {file} the file name · {date} today. "
                      "Double-click a template to edit it; drag to re-order.")
        info.setWordWrap(True)
        apply_style(info, "color:@muted;")
        v.addWidget(info)
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        for s in ST.load_templates(settings):
            self._add(s)
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(newButton("Add", slot=self._new))
        row.addWidget(newButton("Remove", slot=self._remove))
        row.addStretch(1)
        row.addWidget(newButton("Reset", slot=self._reset, tip="The built-in templates"))
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _add(self, text):
        it = QListWidgetItem(text)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(it)
        return it

    def _new(self):
        it = self._add("New text")
        self.list.setCurrentItem(it)
        self.list.editItem(it)

    def _remove(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _reset(self):
        self.list.clear()
        for s in ST.DEFAULT_TEMPLATES:
            self._add(s)

    def items(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def accept(self):
        ST.save_templates(self.st, self.items())
        super().accept()


class TextDialog(QDialog):
    """Text for the image: click a template (its fields filled in) or type; the size in image pixels."""

    def __init__(self, settings, parent=None, text: str = "", size: int | None = None, fields: dict | None = None,
                 title: str = "Text", size_label: str = "Size:", allow_empty: bool = False, look: str | None = None,
                 image: QImage | None = None, scale: float = 1.0, point=None, sketch_color: str = "#ff3b30",
                 preview_size: float | None = None, fixed_look: str | None = None):
        """look: show the look picker (key of the current one). image / scale / point: the picture the text goes on
        (shown at scale × the image), and where (image px) — for the preview and the "auto" colours.
        preview_size: the text size to preview with when there is no size box (auto-sketch); fixed_look: the look
        to preview with when it is chosen elsewhere (auto-sketch: the batch's)."""
        super().__init__(parent)
        self.st, self.fields = settings, dict(fields or {})
        self.allow_empty = allow_empty
        self.image, self.scale, self.point = image, scale, point
        self.sketch_color = sketch_color
        self.preview_size = preview_size
        self.fixed_look = fixed_look
        self.setWindowTitle(title)
        self.resize(560, 560)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Templates — click one to use it:"))
        self.templates = QListWidget()
        self.templates.setMaximumHeight(170)
        self.templates.itemClicked.connect(self._use)
        self.templates.itemActivated.connect(lambda it: (self._use(it), self.accept()))
        v.addWidget(self.templates)
        self._fill()
        self.edit = QPlainTextEdit(text)
        self.edit.setPlaceholderText("Text to put on the image")
        self.edit.setMaximumHeight(90)
        v.addWidget(self.edit)
        row = QHBoxLayout()
        self.size = None
        if size is not None:
            row.addWidget(QLabel(size_label))
            self.size = QSpinBox()
            self.size.setRange(6, 600)
            self.size.setSuffix(" px")
            self.size.setValue(int(size))
            self.size.setToolTip("Height of the letters in image pixels")
            row.addWidget(self.size)
        self.look_box = None
        if look is not None:
            row.addSpacing(10)
            row.addWidget(QLabel("Look:"))
            self.look_box = QComboBox()
            for k, name, *_r in ST.LOOKS:
                self.look_box.addItem(name, k)
            self.look_box.setCurrentIndex(max(0, self.look_box.findData(look)))
            self.look_box.setToolTip("Style and colours. “for the background”: the colour is chosen to stand out from "
                                     "the part of the image the text is put on")
            row.addWidget(self.look_box, 1)
        row.addStretch(0 if look is not None else 1)
        row.addWidget(newButton("Edit templates…", slot=self._edit_templates))
        v.addLayout(row)
        self.preview = TextPreview()
        v.addWidget(self.preview, 1)
        self.preview.set_background(image, scale)
        self.edit.textChanged.connect(self._refresh)
        if self.size is not None:
            self.size.valueChanged.connect(self._refresh)
        if self.look_box is not None:
            self.look_box.currentIndexChanged.connect(self._refresh)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.edit.setFocus()
        self._refresh()

    def styled_el(self) -> dict | None:
        """The text element exactly as it will be drawn (None while there is no text)."""
        from .sketch import styled_text_el
        txt = self.text()
        if not txt:
            return None
        fs = self.value() or self.preview_size or 40
        pt = self.point
        if pt is None:
            pt = (fs * 0.5, fs * 1.5)
        from .sketch import fit_inside
        el = styled_text_el(txt, pt, fs, self.look() or ST.DEFAULT_LOOK, self.sketch_color, self.image, self.scale)
        if self.image is not None and not self.image.isNull():
            W, H = self.image.width() / self.scale, self.image.height() / self.scale
            moved = fit_inside(el, W, H)                       # never cut off at the image edge
            if moved is not el:                                # its colours belong to where it ends up
                el = styled_text_el(txt, moved["pts"][0], fs, self.look() or ST.DEFAULT_LOOK, self.sketch_color,
                                    self.image, self.scale)
        return el

    def _refresh(self, *_a):
        self.preview.set_el(self.styled_el())

    def look(self) -> str | None:
        return self.look_box.currentData() if self.look_box is not None else self.fixed_look

    def _fill(self):
        self.templates.clear()
        for s in ST.load_templates(self.st):
            it = QListWidgetItem(s)
            it.setData(Qt.ItemDataRole.UserRole, s)
            shown = ST.expand(s, **self.fields)
            if shown != s:                                # the fields filled in for this click
                it.setText(f"{s}    →  {shown}" if shown else s)
                it.setToolTip(f"→ {shown}" if shown else "(no shape here to take {label} / {n} from)")
            self.templates.addItem(it)

    def _use(self, it):
        self.edit.setPlainText(ST.expand(it.data(Qt.ItemDataRole.UserRole), **self.fields))
        self.edit.setFocus()
        self.edit.moveCursor(self.edit.textCursor().MoveOperation.End)

    def _edit_templates(self):
        if TemplateEditor(self.st, self).exec():
            self._fill()

    def text(self) -> str:
        return self.edit.toPlainText().strip()

    def value(self) -> int | None:
        return self.size.value() if self.size is not None else None

    def accept(self):
        if not self.text() and not self.allow_empty:
            self.edit.setFocus()
            return
        super().accept()
