# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Testing (Model Lab): module-level helpers shared by its modules (also re-exported by AIO_QC.widgets.testing.model_lab)."""
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QScrollArea, QFrame, QDialog, QDialogButtonBox,
)
from AIO_QC.widgets.testing.lab_common import (
    Job,
    theme_mode,
    bgr_pixmap,
)


def _release_thread_reader():
    from AIO_QC.core.testing.pipeline import release_reader
    release_reader()
if _release_thread_reader not in Job.thread_cleanups:
    Job.thread_cleanups.append(_release_thread_reader)
T_OVERVIEW, T_CHARTS, T_CONFUSION, T_CLASSES, T_FAILURES, T_PLOTS, T_COMPARE, T_VIDEOS = range(8)
STEPS = [("Test data", "Videos, image folders, labels"), ("Models", "Weights to test"),
         ("Run", "Predict, cached & resumable"), ("Results", "Charts, plots, failures"),
         ("Report", "HTML report & data"), ("Review videos", "Play videos with every model's detections")]
def _shared_env():
    """The one environment for .pt / .engine models (Image QC ▸ Models & Environment)."""
    from AIO_QC.utils.qt_utils import app_settings
    return app_settings().value("inference/python", "") or ""
def _set_shared_env(path):
    if path:
        from AIO_QC.utils.qt_utils import app_settings
        app_settings().setValue("inference/python", path)
def _unique_tip(k):
    from AIO_QC.tools.frame_extractor.core.dedup import PRESET_HELP
    return "Keep every sampled frame." if k == "off" else PRESET_HELP.get(k, "")
def slot_hex(slot, mode=None):
    from AIO_QC.tools.frame_extractor.core.common import model_hex
    return model_hex(slot, mode or theme_mode())
def _swatch_icon(color, size=12):
    from AIO_QC.utils import painted_icon
    c = QColor(color)
    return painted_icon(lambda p: p.fillRect(QRectF(0, 0, size, size), c), size)
def _item(text, sort=None, align_right=False, tip=None):
    it = QTableWidgetItem(str(text))
    if sort is not None:
        it.setData(Qt.ItemDataRole.UserRole, sort)
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it
class NumItem(QTableWidgetItem):
    """Sorts by its numeric value (UserRole), not its text."""

    def __lt__(self, other):
        a, b = self.data(Qt.ItemDataRole.UserRole), other.data(Qt.ItemDataRole.UserRole)
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return super().__lt__(other)
def _num_item(v, fmt="{:.3f}", tip=None):
    it = NumItem("–" if v is None else fmt.format(v))
    it.setData(Qt.ItemDataRole.UserRole, -1e18 if v is None else float(v))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    return it
def _make_table(headers, stretch_col=None):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_col is not None:
        hh.setSectionResizeMode(stretch_col, QHeaderView.ResizeMode.Stretch)
    return t
def _status_icon(level, size=14):
    """Step status for the navigation: ✓ ok · ! warning · ✕ problem · ○ not done yet."""
    from AIO_QC.utils import painted_icon
    col = {"ok": "#199e70", "ready": "#199e70", "warn": "#c98500", "error": "#d03b3b"}.get(level, "#8f8d86")
    return painted_icon(lambda p: _draw_status(p, level, size, col), size)
def _draw_status(p, level, size, col):
    from PyQt6.QtGui import QPen
    from PyQt6.QtCore import QRectF
    r = QRectF(1, 1, size - 2, size - 2)
    if level == "todo":
        p.setPen(QPen(QColor(col), 1.5))
        p.drawEllipse(r)
    else:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(col))
        p.drawEllipse(r)
        p.setPen(QPen(QColor("white"), 1.8))
        c = size / 2
        if level in ("ok", "ready"):
            p.drawPolyline([QPointF(c - 3, c), QPointF(c - 0.8, c + 2.4), QPointF(c + 3.4, c - 2.6)])
        elif level == "warn":
            p.drawLine(QPointF(c, c - 3.2), QPointF(c, c + 0.8))
            p.drawPoint(QPointF(c, c + 3))
        else:
            p.drawLine(QPointF(c - 2.6, c - 2.6), QPointF(c + 2.6, c + 2.6))
            p.drawLine(QPointF(c + 2.6, c - 2.6), QPointF(c - 2.6, c + 2.6))
def _strip_tags(t):
    import re
    return re.sub(r"<[^>]+>", "", t)
def _hint(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setEnabled(False)           # muted in both themes
    return lab
def _scroll(widget):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(widget)
    return sa
class ImagePreview(QDialog):
    """Large view of one result image, with actions."""

    def __init__(self, parent, img, title, actions=()):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        scr = parent.screen().availableGeometry() if parent is not None and parent.screen() else None
        mw = int(scr.width() * 0.8) if scr else 1400
        mh = int(scr.height() * 0.75) if scr else 800
        lab = QLabel()
        lab.setPixmap(bgr_pixmap(img, mw, mh))
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lab, 1)
        bb = QDialogButtonBox()
        for text, fn in actions:
            b = bb.addButton(text, QDialogButtonBox.ButtonRole.ActionRole)
            b.clicked.connect(fn)
        bb.addButton(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        lay.addWidget(bb)

