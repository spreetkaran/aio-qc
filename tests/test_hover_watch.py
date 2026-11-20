# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Hover opens the QC drawer and unfolds the image details even when Qt's Enter / Leave events and
QApplication.mouseButtons() are stale (Windows: after the native folder dialog, a drag and drop, another active
window) — the cursor position and the buttons of the real mouse moves decide."""
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QCursor, QMouseEvent
from PyQt6.QtWidgets import QApplication

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _move_elsewhere(target, gp):
    """A real mouse move that Qt routes to some other widget (no Enter for the one under the cursor)."""
    QCursor.setPos(gp)
    lp = QPointF(target.mapFromGlobal(gp))
    ev = QMouseEvent(QEvent.Type.MouseMove, lp, QPointF(gp), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, ev)


def test_hover_works_with_stale_qt_state(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.widgets.canvas_hud import _InfoChip
    from AIO_QC.widgets.ribbon import Ribbon
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.window().show()
    qtbot.wait(300)
    rb, chip = w.findChild(Ribbon), w.findChild(_InfoChip)
    assert rb is not None and chip is not None and chip.isVisible()
    rb.set_pinned(False) if rb.pinned else None
    # Qt still believes a button is down (the click the folder dialog consumed)
    monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.MouseButton.LeftButton))
    tab = rb.bar.tabRect(1).center()
    _move_elsewhere(w.canvas.viewport(), rb.bar.mapToGlobal(tab))
    qtbot.waitUntil(lambda: rb._open, timeout=3000)                    # opened on hover, no click
    rb.close_drop(animate=False)
    qtbot.wait(50)
    _move_elsewhere(w.canvas.viewport(), chip.mapToGlobal(chip.rect().center()))
    qtbot.waitUntil(lambda: chip._open > 0.99, timeout=3000)            # details unfold
    far = w.canvas.viewport().mapToGlobal(w.canvas.viewport().rect().center())
    _move_elsewhere(w.canvas.viewport(), far)
    qtbot.waitUntil(lambda: chip._open < 0.01, timeout=3000)            # and fold back
