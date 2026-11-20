# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Windows sized by the screen they open on; rows of buttons wrap instead of forcing a window wide; parameter forms
give way on short screens. (Measured across 1280×720 … 3840×2160 with a probe that opens every window.)"""
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from AIO_QC.utils.qt_utils import WINDOW_SIZES, FlowLayout, SplitRow, fit_size, flow_row, wrap_long_labels
from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_flow_layout_wraps_and_needs_only_its_widest_widget(qtbot):
    host = QWidget()
    flow = FlowLayout(host, spacing=6)
    btns = [QPushButton(f"Action number {i}") for i in range(8)]
    for b in btns:
        flow.addWidget(b)
    qtbot.addWidget(host)
    widest = max(b.sizeHint().width() for b in btns)
    assert flow.minimumSize().width() == widest                     # not the sum of all of them
    one_row = flow.heightForWidth(sum(b.sizeHint().width() + 6 for b in btns))
    assert flow.heightForWidth(widest + 10) > 3 * one_row           # narrow: several rows
    row = flow_row([("conf ≥", QPushButton("0.25")), QPushButton("Go")])
    assert row.layout().count() == 2                                # a label + its box stay one unit


def test_split_row_needs_the_wider_group_and_wraps(qtbot):
    host, a, b = QWidget(), QPushButton("left " * 8), QPushButton("right " * 6)
    lay = SplitRow(a, b, host)
    qtbot.addWidget(host)
    assert lay.minimumSize().width() == max(a.minimumSizeHint().width(), b.minimumSizeHint().width())
    wide = a.sizeHint().width() + b.sizeHint().width() + 40
    assert lay.heightForWidth(wide) < lay.heightForWidth(a.sizeHint().width())   # one line, then two
    from PyQt6.QtCore import QRect
    lay.setGeometry(QRect(0, 0, wide, lay.heightForWidth(wide)))
    assert b.geometry().right() == wide - 1 and a.geometry().top() == b.geometry().top() or \
        abs(a.geometry().center().y() - b.geometry().center().y()) <= 1             # right group at the right edge
    narrow = a.sizeHint().width()
    lay.setGeometry(QRect(0, 0, narrow, lay.heightForWidth(narrow)))
    assert b.geometry().top() > a.geometry().bottom() and b.geometry().right() == narrow - 1


def test_fit_size_follows_the_screen_and_the_content(qtbot):
    g = QApplication.primaryScreen().availableGeometry()
    for kind, (fw, fh, cw, ch) in WINDOW_SIZES.items():
        d = QDialog()
        QVBoxLayout(d).addWidget(QLabel("x"))
        qtbot.addWidget(d)
        fit_size(d, kind)
        assert d.width() <= int(g.width() * 0.96) and d.height() <= int(g.height() * 0.94)
        assert d.width() >= min(int(g.width() * fw), cw, int(g.width() * 0.96)) - 1
    big = QDialog()
    lab = QLabel("word " * 400)                                     # one very long line
    QVBoxLayout(big).addWidget(lab)
    qtbot.addWidget(big)
    wrap_long_labels(big)
    assert lab.wordWrap()


def test_model_check_actions_wrap_and_params_give_way(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.qc_actions["model_check"].trigger()
    qtbot.waitUntil(lambda: "model_check" in w.open_dialogs, timeout=3000)
    d = w.open_dialogs["model_check"]
    assert isinstance(d.actions_layout, FlowLayout)
    assert d.params_scroll.widget() is d.params_box and d.params_scroll.minimumHeight() <= 90
    d.params_toggle.setChecked(False)
    assert not d.params_scroll.isVisible()
    d.params_toggle.setChecked(True)
    assert d.params_scroll.isVisible()
    assert d.minimumSizeHint().width() < 1100                       # was 1670: one row of every button
