# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""No widget flashes up as a window of its own while the UI is built (on Windows each flash takes the focus — the
console's selection flickered before the app opened), and background loops wait while the view animates."""
import time
import weakref

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QWidget

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


class _Stray(QObject):
    def __init__(self):
        super().__init__()
        self.shown = []

    def eventFilter(self, o, e):
        if e.type() == QEvent.Type.Show and isinstance(o, QWidget) and o.isWindow() and o.parent() is None:
            self.shown.append((weakref.ref(o), type(o).__name__))
        return False

    def strays(self):
        out = []
        for ref, name in self.shown:
            o = ref()
            try:
                if o is not None and o.parent() is not None:
                    out.append(name)
            except RuntimeError:
                pass
        return out


def test_no_stray_windows_while_building(qapp, qtbot, tmp_path, shell):  # noqa: F811
    f = _Stray()
    qapp.installEventFilter(f)
    found = []
    try:
        w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
        found += f.strays()
        for key in ("validation", "stats", "outliers", "crop_gallery"):
            w.qc_actions[key].trigger()
            qtbot.waitUntil(lambda: key in w.open_dialogs, timeout=3000)
            found += [f"{key}: {n}" for n in f.strays()]           # checked while the window is still alive
            f.shown.clear()
            w.open_dialogs[key].close()
        for mode in ("data", "video", "image"):
            shell.switch(mode)
            found += [f"{mode}: {n}" for n in f.strays()]
            f.shown.clear()
    finally:
        qapp.removeEventFilter(f)
    assert found == []


def test_background_loops_wait_for_the_view():
    from AIO_QC.core import ui_yield
    ui_yield.interacting(0.2)
    assert ui_yield.busy()
    t0 = time.monotonic()
    ui_yield.yield_to_ui()
    assert 0.1 < time.monotonic() - t0 < 1.0 and not ui_yield.busy()
    t0 = time.monotonic()
    ui_yield.yield_to_ui()                                   # idle: no wait at all
    assert time.monotonic() - t0 < 0.01
    ui_yield.interacting(5)
    t0 = time.monotonic()
    ui_yield.yield_to_ui(stop=lambda: True)                  # a stopping worker never waits
    assert time.monotonic() - t0 < 0.05
    ui_yield._until = 0.0


def test_eye_toggle_uses_the_style_sheet_hover(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QHBoxLayout, QToolButton
    from AIO_QC.widgets import panels
    w = QWidget()
    lay = QHBoxLayout(w)
    eye, ref = panels.EyeToggle("a", "b"), QToolButton()
    ref.setAutoRaise(True)
    ref.setFixedSize(24, 24)
    for b in (eye, ref):
        lay.addWidget(b)
    w.show()
    qapp.processEvents()
    for b in (eye, ref):
        b.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
    img_e, img_r = eye.grab().toImage(), ref.grab().toImage()
    assert img_e.pixelColor(3, 12) == img_r.pixelColor(3, 12)     # same hover tone at the edge (no eye there)
    w.close()
