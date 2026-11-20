# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Feature windows (every mode's tool windows, not the main window) have rounded corners; maximised ones are square."""
from PyQt6.QtCore import QPoint

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_feature_windows_round_main_window_not(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.qc_actions["validation"].trigger()
    qtbot.waitUntil(lambda: "validation" in w.open_dialogs, timeout=3000)
    d = w.open_dialogs["validation"]
    qtbot.waitUntil(d.isVisible, timeout=3000)
    d.resize(700, 500)
    qtbot.wait(30)
    assert d._aio_chrome.corners in ("mask", "dwm")
    if d._aio_chrome.corners == "mask":
        m = d.mask()
        assert not m.contains(QPoint(0, 0)) and not m.contains(QPoint(d.width() - 1, d.height() - 1))   # corners cut
        assert m.contains(QPoint(d.width() // 2, 0)) and m.contains(QPoint(0, d.height() // 2))           # edges kept
        assert m.boundingRect().size() == d.size()                                                       # follows resizes
    d.showMaximized()
    qtbot.waitUntil(lambda: d._aio_chrome.corners == "square", timeout=3000)
    assert d.mask().isEmpty()
    d.showNormal()
    qtbot.waitUntil(lambda: d._aio_chrome.corners in ("mask", "dwm"), timeout=3000)
    d.close()
    assert shell.host.mask().isEmpty()                                  # the main window frame is untouched
