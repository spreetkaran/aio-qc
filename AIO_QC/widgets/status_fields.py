# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Paint-style status bar: pointer position · selection size · image size on the left (icon + value, fixed slots so
nothing jumps), then the temporary messages; the app's permanent indicators stay on the right.

QStatusBar hides its normal widgets while it shows a message — here the fields are part of one permanent,
stretching strip and the message is drawn in that strip, so the fields are always visible.
"""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QStatusBar, QStyle, QStyleOption, QToolButton,
                             QWidget)

from AIO_QC.utils import set_icon, style as apply_style


class _Elided(QLabel):
    """One line, elided to the width it has now (re-elides on resize); full text in the tooltip when cut."""

    def __init__(self):
        super().__init__("")
        self._full = ""

    def setText(self, text):
        self._full = text or ""
        super().setText(self._full)
        self._fit()

    def text(self):
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        fm = self.fontMetrics()
        cut = fm.horizontalAdvance(self._full) > self.width() > 20
        QLabel.setText(self, fm.elidedText(self._full, Qt.TextElideMode.ElideRight, self.width()) if cut
                       else self._full)
        self.setToolTip(self._full if cut else "")


def file_type(path=None, fmt: str | None = None) -> str:
    """The file's own extension in capitals, as Windows Photos shows it: 'photo.jpg' -> 'JPG', 'scan.jpeg' -> 'JPEG',
    'a.tif' -> 'TIF'. Embedded images (no file) show their decoded format."""
    ext = (str(path).rsplit(".", 1)[-1] if path and "." in str(path).replace("\\", "/").rsplit("/", 1)[-1]
           else (fmt or ""))
    return ext.upper()


def human_size(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,} bytes" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


class Field(QWidget):
    def __init__(self, icon: str | None, tip: str, width: int):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(8)
        self.icon = QToolButton()
        self.icon.setAutoRaise(True)
        self.icon.setIconSize(QSize(16, 16))
        self.icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        apply_style(self.icon, "QToolButton{border:none; padding:0;}")
        if icon:
            set_icon(self.icon, icon)
        self.text = QLabel("")
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if icon:
            lay.addWidget(self.icon)
        lay.addWidget(self.text, 1)
        self.setFixedWidth(width)
        self.setToolTip(tip)
        self._tip = tip

    def set(self, text: str, tip: str | None = None):
        if self.text.text() != text:
            self.text.setText(text)
        self.setToolTip(tip or self._tip)


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1)
    apply_style(f, "QFrame{color:@line; background:@line; border:none; margin:4px 0;}")
    return f


class PaintStatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeGripEnabled(True)
        self.strip = QWidget()
        lay = QHBoxLayout(self.strip)
        lay.setContentsMargins(4, 0, 0, 0)
        lay.setSpacing(4)
        self.pointer = Field("status-pointer", "Pointer position on the image (pixels)", 150)
        self.selection = Field("status-selection", "Size of the selected shape, the box being drawn or the ROI", 160)
        self.image = Field("status-image-size", "Image size (pixels)", 170)
        self.ftype = Field(None, "File type of the image", 70)
        self.fsize = Field("status-file-size", "Size of the image file", 130)
        for w in (self.pointer, _sep(), self.selection, _sep(), self.image, _sep(), self.ftype, _sep(), self.fsize,
                  _sep()):
            lay.addWidget(w)
        self.message = _Elided()
        self.message.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        apply_style(self.message, "padding-left:6px;")
        lay.addWidget(self.message, 1)
        self.addPermanentWidget(self.strip, 1)
        self.messageChanged.connect(self._message)

    def _message(self, text):
        self.message.setText(text)

    def paintEvent(self, _e):
        """Background only: the message is shown in the strip (QStatusBar would paint it over the fields)."""
        p = QPainter(self)
        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_PanelStatusBar, opt, p, self)

    # ---- values
    def set_pointer(self, pt):
        self.pointer.set("" if pt is None else f"{int(pt.x())}, {int(pt.y())}px")

    def set_selection(self, rect):
        if rect is None or rect.width() <= 0 and rect.height() <= 0:
            self.selection.set("")
            return
        self.selection.set(f"{rect.width():.0f} × {rect.height():.0f}px",
                           f"x {rect.left():.1f} – {rect.right():.1f} · y {rect.top():.1f} – {rect.bottom():.1f}")

    def set_image(self, w: int | None, h: int | None):
        self.image.set("" if not w or not h else f"{int(w)} × {int(h)}px")

    def set_file(self, path=None, size: int | None = None, fmt: str | None = None):
        """Type and size of the shown image — only once it decoded (None clears both)."""
        if path is None and size is None:
            self.ftype.set("")
            self.fsize.set("")
            return
        if size is None and path is not None:
            import os
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
        self.ftype.set(file_type(path, fmt))
        self.fsize.set(human_size(size), f"{size:,} bytes" if size is not None else None)
