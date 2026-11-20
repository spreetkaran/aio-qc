# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Mode windows in the background: what changes meanwhile must look right when they come back to the front.

(isVisible() is False for every widget of a hidden window, so state logic based on it silently skips updates.)"""
from __future__ import annotations

from PyQt6.QtWidgets import QAbstractButton, QWidget

from .test_modes import shell3  # noqa: F401


def _pump(qtbot, ms=40):
    qtbot.wait(ms)


def _switch(sh, mode):
    """Bring a mode to the front the way the mode switch does (the others are hidden)."""
    if sh.mode is None:
        return sh.show(mode)
    sh.switch(mode)
    return sh.window(mode)


def test_one_settings_gear_per_window(shell3, qtbot):  # noqa: F811
    """The Frame Extractor installs its gear, then the shell adds the mode switch: one gear, next to the switch,
    never a second one left over the File menu."""
    for mode in ("image", "video", "data"):
        win = shell3.show(mode)
        qtbot.addWidget(win)
        _pump(qtbot)
        mb = win.menuBar()
        boxes = mb.findChildren(QWidget, "settingsCorner")
        gears = [b for b in mb.findChildren(QAbstractButton, "settingsButton") if b.isVisibleTo(win)]
        assert len(boxes) == 1 and len(gears) == 1, mode
        assert mb.cornerWidget() is boxes[0]
        assert win.mode_switch.parent() is boxes[0]
        gear_x = gears[0].mapTo(mb, gears[0].rect().topLeft()).x()
        file_menu = mb.actionGeometry(mb.actions()[0])
        assert gear_x > file_menu.right(), mode                       # right side, not over "File"


def test_task_indicator_clears_in_background_windows(shell3, qtbot):  # noqa: F811
    from AIO_QC.widgets import tasks
    img = _switch(shell3, "image")
    qtbot.addWidget(img)
    ind = img._task_indicator
    t = tasks.start("Getting libmpv · smoother playback")
    _pump(qtbot, 150)
    assert not ind.isHidden() and "libmpv" in ind.text()
    vid = _switch(shell3, "video")                                        # Image QC goes to the background …
    qtbot.addWidget(vid)
    _pump(qtbot)
    assert not vid._task_indicator.isHidden()                         # the same job shows there too
    t.finish()                                                         # … the job ends meanwhile
    _pump(qtbot, 150)
    assert "Done" in vid._task_indicator.text()                       # the green tick for a moment …
    qtbot.waitUntil(lambda: vid._task_indicator.isHidden(), timeout=4000)     # … then it goes
    _switch(shell3, "image")
    _pump(qtbot)
    assert ind.isHidden() and not ind.isVisible()                     # nothing stale when it comes back
    # and a job that starts while a window is in the background shows once it is in front
    _switch(shell3, "video")
    t2 = tasks.start("Batch extraction")
    _pump(qtbot, 150)
    _switch(shell3, "image")
    _pump(qtbot)
    assert ind.isVisible() and "Batch extraction" in ind.text()
    t2.finish()
    qtbot.waitUntil(lambda: ind.isHidden(), timeout=4000)


def test_toast_posted_in_background_counts_down(shell3, qtbot):  # noqa: F811
    from PyQt6.QtCore import QVariantAnimation
    img = _switch(shell3, "image")
    qtbot.addWidget(img)
    vid = _switch(shell3, "video")
    qtbot.addWidget(vid)
    _pump(qtbot)
    t = img.toast.show_message("3 files changed outside", timeout_ms=4000)
    _pump(qtbot, 100)
    assert t.count.state() != QVariantAnimation.State.Running        # not seen yet: not counting
    _switch(shell3, "image")
    qtbot.waitUntil(lambda: t.count.state() == QVariantAnimation.State.Running, timeout=3000)   # seen: counts down
    sb = img.statusBar()
    qtbot.waitUntil(lambda: sb.isHidden() and t.geometry().bottom() <= img.height()     # the welcome page has no bar
                    or t.geometry().bottom() <= sb.geometry().top(), timeout=3000)     # above the status bar


def test_dock_toggle_while_window_hidden(shell3, qtbot):  # noqa: F811
    from AIO_QC.widgets import anim
    img = _switch(shell3, "image")
    qtbot.addWidget(img)
    dock = img.filmstrip_dock
    anim.toggle_dock(img, dock, True)
    _pump(qtbot, 300)
    vid = _switch(shell3, "video")
    qtbot.addWidget(vid)
    _pump(qtbot)
    anim.toggle_dock(img, dock, False)                                # e.g. a setting changed elsewhere
    _pump(qtbot, 300)
    _switch(shell3, "image")
    _pump(qtbot)
    assert dock.isHidden()


def test_frame_extractor_jobs_take_the_output_line_place(shell3, qtbot):  # noqa: F811
    """Frame Extractor: a background job shows at the far left, in place of the "Output Dir" line; the readouts to the
    right do not move; when the job is done the output line comes back."""
    from AIO_QC.widgets import tasks
    vid = _switch(shell3, "video")
    qtbot.addWidget(vid)
    vid.resize(1400, 800)
    _pump(qtbot, 80)
    ind = vid._task_indicator
    assert vid.status_text().startswith("Output Dir") and not vid.msg_label.isHidden() and ind.isHidden()
    right = [w for w in (vid.pixel_label, vid.decoder_label, vid.render_label)]
    xs = [w.mapTo(vid, w.rect().topLeft()).x() for w in right]
    t = tasks.start("Timeline analysis · test.mp4")
    t.update(0.4, "syncing")
    qtbot.waitUntil(lambda: not ind.isHidden(), timeout=2000)
    _pump(qtbot, 80)
    assert vid.msg_label.isHidden()                                       # the job is in the output line's place
    assert ind.mapTo(vid, ind.rect().topLeft()).x() <= vid.msg_label.mapTo(vid, vid.msg_label.rect().topLeft()).x() + 2
    assert [w.mapTo(vid, w.rect().topLeft()).x() for w in right] == xs     # nothing to the right moved
    vid.flash_status("Saved frame", 200)                                  # a short note shows over it for a moment
    assert not vid.msg_label.isHidden() and ind.isHidden()
    qtbot.waitUntil(lambda: not ind.isHidden(), timeout=2000)
    t.finish()
    qtbot.waitUntil(lambda: ind.isHidden() and not vid.msg_label.isHidden(), timeout=5000)
    assert vid.status_text().startswith("Output Dir")
    assert [w.mapTo(vid, w.rect().topLeft()).x() for w in right] == xs


def test_fx_left_slot_keeps_the_job_through_hints_and_errors(shell3, qtbot):  # noqa: F811
    """A button's hint (status tip) or an error note shows in the left slot for a moment; Qt's own handling would
    hide the job (and the output line) instead."""
    from PyQt6.QtGui import QStatusTipEvent
    from PyQt6.QtWidgets import QApplication
    from AIO_QC.widgets import tasks
    vid = _switch(shell3, "video")
    qtbot.addWidget(vid)
    vid.resize(1400, 800)
    _pump(qtbot, 80)
    ind = vid._task_indicator
    t = tasks.start("Timeline analysis · x.mp4")
    qtbot.waitUntil(lambda: not ind.isHidden(), timeout=2000)
    QApplication.sendEvent(vid, QStatusTipEvent("Play / pause"))
    assert not vid.msg_label.isHidden() and "Play / pause" in vid.status_text() and ind.isHidden()
    QApplication.sendEvent(vid, QStatusTipEvent(""))
    assert not ind.isHidden() and vid.msg_label.isHidden()
    shell3.log_errors(["Traceback", "ValueError: boom"])
    assert "boom" in vid.status_text() and not vid.msg_label.isHidden()
    vid.reset_status_priority()
    assert not ind.isHidden()
    t.finish()
    qtbot.waitUntil(lambda: ind.isHidden() and not vid.msg_label.isHidden(), timeout=5000)
    assert vid.status_text().startswith("Output Dir")


def test_frame_extractor_fits_a_laptop_screen(shell3, qtbot):  # noqa: F811
    """The Frame Extractor's side panels and transport row give way: the window is never forced wider than a
    1280 px screen (it needed 1545 px), the step / speed / zoom group moves under the time readout instead, and
    menus or tools that don't fit sit behind a visible chevron."""
    from PyQt6.QtWidgets import QMenuBar, QToolButton
    vid = _switch(shell3, "video")
    qtbot.addWidget(vid)
    vid.central_stack.setCurrentIndex(0)                 # the player (no video: the start page is in its place)
    assert vid.minimumSizeHint().width() < 1000
    row = vid.transport_row
    left, right = (row.layout().itemAt(i).widget() for i in range(2))
    vid.window().resize(1900, 800)
    _pump(qtbot, 80)
    one = row.height()
    assert abs(left.geometry().top() - right.geometry().top()) < 30           # one line on a wide window
    vid.window().resize(900, 700)
    _pump(qtbot, 80)
    assert right.geometry().top() > left.geometry().bottom() - 2              # two lines on a narrow one
    assert row.height() > one and right.geometry().right() <= row.width()
    ext = vid.toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
    assert ext is not None and not ext.icon().isNull()
    mext = vid.findChild(QMenuBar).findChild(QToolButton, "qt_menubar_ext_button")
    assert mext is not None and not mext.icon().isNull()
