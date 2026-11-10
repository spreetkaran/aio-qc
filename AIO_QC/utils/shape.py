# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor
from .theme import style as apply_style
from PyQt6.QtWidgets import QFrame, QGraphicsDropShadowEffect


def custom_v_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Plain)
    apply_style(sep, "color: @muted; margin-left: 6px; margin-right: 6px;")
    sep.setFixedHeight(22)
    return sep


def get_color_icon(color):
    """A colour dot drawn at the real screen resolution (sharp at any scaling)."""
    from .qt_utils import painted_icon
    c = QColor(color)

    def draw(p):
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(0.5, 0.5, 11, 11))
    return painted_icon(draw, 12)


class ShapeGroup:
    """Graphics items of one annotation: outline + tag background + tag text."""

    def __init__(self, index: int, shape_data: dict, color: QColor, shape_item, rect_item=None, text_item=None):
        self.index = index            # index into the annotation's shapes list
        self.shape_data = shape_data
        self.color = color
        self.shape_item = shape_item
        self.rect_item = rect_item
        self.text_item = text_item
        self.base_z = {id(i): i.zValue() for i in self.items()}
        self.base_width = shape_item.pen().widthF()
        self.base_brush = shape_item.brush() if hasattr(shape_item, "brush") else None
        self.selected = False
        self.hovered = False

    def items(self):
        return [i for i in (self.shape_item, self.rect_item, self.text_item) if i is not None]

    def set_hover(self, on: bool):
        """Pre-click feedback: thicker, lighter outline + faint fill. Selection styling always wins."""
        self.hovered = on
        if self.selected:
            return
        pen = self.shape_item.pen()
        pen.setWidthF(self.base_width * (1.7 if on else 1.0))
        pen.setColor(self.color.lighter(135) if on else self.color)
        self.shape_item.setPen(pen)
        if self.base_brush is not None:
            fill = QColor(self.color)
            fill.setAlpha(45)
            self.shape_item.setBrush(fill if on else self.base_brush)

    def set_selected(self, selected: bool, top_z: float = 1e9):
        self.selected = selected
        pen = self.shape_item.pen()
        pen.setWidthF(self.base_width * 2 if selected else self.base_width)
        pen.setColor(self.color.lighter(150) if selected else self.color)
        pen.setStyle(Qt.PenStyle.SolidLine)
        self.shape_item.setPen(pen)
        if self.base_brush is not None:
            fill = QColor(self.color)
            fill.setAlpha(90)
            self.shape_item.setBrush(fill if selected else self.base_brush)
        effect = None
        if selected:
            effect = QGraphicsDropShadowEffect()
            effect.setBlurRadius(18)
            effect.setColor(self.color)
            effect.setOffset(0)
        # A QGraphicsEffect can only be owned by one item -> only the outline glows.
        self.shape_item.setGraphicsEffect(effect)
        for n, item in enumerate(self.items()):
            item.setZValue(top_z + n if selected else self.base_z[id(item)])
        if not selected and self.hovered:
            self.set_hover(True)          # still under the cursor: fall back to the hover look
