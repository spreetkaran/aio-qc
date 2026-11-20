# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Minimise, from the user's side: a feature window's minimise button tucks it into the shelf (its job keeps
running, the chip shows the progress, then Done); a chip click or opening the feature again brings it back where it
was; anything else that minimises a feature window (Win+Down, a desktop's own button) also ends in the shelf; Testing
and the Crop Viewer come back from the menu; × on a chip closes the window (asking while a job runs); a job that ends
while the app is minimised asks for attention; modal dialogs and fixed pop-ups have no minimise."""
from PyQt6.QtCore import Qt

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _chips(w):
    s = getattr(w.window(), "_aio_shelf", None)
    return list(s.chips.values()) if s is not None else []


def test_shelf_minimise_bring_back_jobs_and_attention(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.widgets import tasks, window_shelf as WS
    w = _open(shell, qtbot, FX.labelme(tmp_path / "lm", n=4))
    w.open_tool("stats")
    d = w.open_dialogs["stats"]
    qtbot.waitUntil(d.isVisible, timeout=5000)
    ctl = d._window_controls
    assert not ctl.min_btn.isHidden() and not ctl.max_btn.isHidden()
    d.resize(420, 300)                                            # inside the test screen (800 × 600)
    d.move(120, 90)
    geo = d.geometry()
    # a job of this window, then the minimise button: hidden, job alive, chip with progress
    job = tasks.start("Stats · 4 files", owner=d)
    job.update(0.4, "counting")
    ctl.min_btn.click()
    assert not d.isVisible() and WS.is_shelved(d)
    (chip,) = _chips(w)
    chip.refresh()
    assert chip.state.text().startswith("40 %") and chip.bar.value() == 400
    assert w.window()._aio_shelf.isVisible()
    # opening the feature again (menu / shortcut) brings it back, where it was
    w.open_tool("stats")
    qtbot.waitUntil(d.isVisible, timeout=5000)
    assert not WS.is_shelved(d) and not _chips(w) and d.geometry() == geo and not d.isMinimized()
    # tucked away again; the job ends: the chip says Done; a click brings it back
    ctl.min_btn.click()
    job.finish()
    (chip,) = _chips(w)
    assert chip.done and chip.state.text().startswith("✓ done")
    chip.clicked.emit()
    assert d.isVisible() and not _chips(w)
    # minimised by anything else (Win+Down, a desktop's own title-bar button): the shelf, never lost
    d.showMinimized()
    qtbot.waitUntil(lambda: WS.is_shelved(d), timeout=3000)
    assert not d.isMinimized() and not d.isVisible()
    w.open_tool("stats")
    qtbot.waitUntil(d.isVisible, timeout=5000)
    assert not d.isMinimized() and not _chips(w)
    # × on a chip closes the window — asks first while its job runs (conftest answers Yes)
    job2 = tasks.start("Stats again", owner=d)
    ctl.min_btn.click()
    _chips(w)[0]._close()
    assert "stats" not in w.open_dialogs and not _chips(w)
    job2.finish()
    # attention when a job ends while the app is minimised
    calls = []
    monkeypatch.setattr(WS.QApplication, "alert", staticmethod(lambda win, ms=0: calls.append(win)))
    monkeypatch.setattr(WS, "ATTENTION_MIN_S", 0.0)
    w.open_tool("dim_area")
    d2 = w.open_dialogs["dim_area"]
    j = tasks.start("Dim / Area", owner=d2)
    w.window().showMinimized()
    j.finish()
    assert calls and calls[-1] is w.window()
    w.window().showNormal()
    calls.clear()
    j3 = tasks.start("quick", owner=d2)
    j3.cancel_fn = None
    j3.cancelled = True
    j3.finish()                                                    # cancelled: no attention
    assert not calls
    d2.close()


def test_testing_and_crop_viewer_come_back(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets import window_shelf as WS
    from AIO_QC.widgets.crop_viewer import open_crop
    w = _open(shell, qtbot, FX.labelme(tmp_path / "lm", n=3))
    lab = w.open_testing()
    qtbot.waitUntil(lab.isVisible, timeout=5000)
    lab._window_controls.min_btn.click()
    assert WS.is_shelved(lab) and not lab.isVisible()
    w.open_testing()
    qtbot.waitUntil(lab.isVisible, timeout=5000)
    assert not lab.isMinimized() and not WS.is_shelved(lab)
    lab.showMinimized()                                          # however it got minimised
    qtbot.waitUntil(lambda: WS.is_shelved(lab), timeout=3000)
    w.open_testing()
    qtbot.waitUntil(lab.isVisible, timeout=5000)
    assert not lab.isMinimized()
    v = open_crop(w, 0)
    qtbot.waitUntil(v.isVisible, timeout=5000)
    v.showMinimized()
    qtbot.waitUntil(lambda: WS.is_shelved(v), timeout=3000)
    open_crop(w, 0)
    assert v.isVisible() and not v.isMinimized() and not WS.is_shelved(v)
    # a window minimised without the shelf (e.g. before the guard): show() / raise_() still bring it back
    v.setProperty("noShelf", True)
    v.showMinimized()
    qtbot.wait(50)
    assert v.isMinimized()
    open_crop(w, 0)
    assert not v.isMinimized()
    lab.close()
    v.close()


def test_no_minimise_for_modal_and_fixed(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
    from AIO_QC.widgets import window_shelf as WS
    w = _open(shell, qtbot, FX.labelme(tmp_path / "lm", n=2))
    m = QDialog(w)
    QVBoxLayout(m).addWidget(QLabel("modal"))
    m.setWindowModality(Qt.WindowModality.WindowModal)
    m.show()
    assert m._window_controls.min_btn.isHidden() and not WS.can_shelve(m)
    m.close()
    f = QDialog(w)
    QVBoxLayout(f).addWidget(QLabel("fixed"))
    f.setFixedSize(300, 120)
    f.show()
    assert f._window_controls.min_btn.isHidden()
    f.close()
