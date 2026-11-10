# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""2.4.0: steady status bar (review mark, position), loading ring + Background Tasks, one menu arrow, centred
pressed buttons, sharp icons instead of emoji, settings reset icon, notifications shrink back after Details."""
from PyQt6.QtWidgets import QMenu, QPushButton, QStyle, QToolButton

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_review_badge_fixed_width_and_no_lag(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None)
    b = w.review_badge
    assert b.text() == "Pending"
    width = b.width()
    first = w.current_key
    w.a.mark_flag.trigger()                         # flags, then opens the next file
    qtbot.waitUntil(lambda: w.current_key != first)
    assert b.text() == "Pending"
    painted = []
    orig = b.repaint
    b.repaint = lambda: (painted.append(b.text()), orig())
    w.a.prev.trigger()                              # back: the mark is painted at once, before the image loads
    qtbot.waitUntil(lambda: w.current_key == first)
    assert b.text() == "Flagged" and painted and set(painted) == {"Flagged"}
    assert b.width() == width and b.minimumWidth() == b.maximumWidth()
    w.session.set_review([first], "flag", "box too loose")
    assert b.note.isVisibleTo(b) and "box too loose" in b.toolTip() and b.width() == width
    # position: as wide as the largest number, so nothing to its left moves
    pw = w.pos_label.minimumWidth()
    w.a.next.trigger()
    qtbot.wait(50)
    assert w.pos_label.minimumWidth() == pw
    assert not w.file_dock.count_label.isVisibleTo(w.file_dock)          # the dock's count is gone


def test_sharp_icons_not_emoji(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_dir_btn.set_status("success")
    assert w.open_dir_btn.status_icon._name == "status-ok"
    w.load_classes_btn.set_status("warning")
    assert w.load_classes_btn.status_icon._name == "status-warning"
    assert "📂" not in w.dataset_label.text() and w.dataset_label.icon._name == "open"
    assert w.mode_label.text() in ("Drag", "Select")


def test_one_menu_arrow_and_centred_press(qapp, shell):  # noqa: F811
    from AIO_QC.widgets.ui_style import AppStyle
    st = qapp.style()
    assert isinstance(st, AppStyle)
    assert st.pixelMetric(QStyle.PixelMetric.PM_ButtonShiftHorizontal) == 0
    assert st.pixelMetric(QStyle.PixelMetric.PM_ButtonShiftVertical) == 0
    a, b = QToolButton(), QToolButton()           # "Add rule ▾" with a menu draws like the same text without one
    for x in (a, b):
        x.setText("Add rule ▾")
        x.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        x.resize(110, 26)
    a.setMenu(QMenu(a))
    assert a.grab().toImage() == b.grab().toImage()
    c, d = QToolButton(), QToolButton()           # without ▾ in the text Qt's own arrow stays (one arrow)
    for x in (c, d):
        x.setText("Add rule")
        x.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        x.resize(110, 26)
    c.setMenu(QMenu(c))
    assert c.grab().toImage() != d.grab().toImage()
    p, q = QPushButton("More ▾"), QPushButton("More ▾")         # with / without a menu: drawn the same
    p.setMenu(QMenu(p))
    for x in (p, q):
        x.resize(120, 30)
    assert p.grab().toImage() == q.grab().toImage()


def test_tasks_ring_tick_and_viewer(shell, qtbot):  # noqa: F811
    from AIO_QC.widgets import tasks
    w = shell.window("image")
    ind = w._task_indicator
    qtbot.waitUntil(lambda: not tasks.center().tasks, timeout=60000)     # earlier tests' jobs (image checks …)
    qtbot.waitUntil(lambda: ind.isHidden(), timeout=4000)
    t = tasks.start("Scanning 10 files", cancel=lambda: None)
    t.update(0.3, "3 / 10")
    qtbot.wait(150)
    assert not ind.isHidden() and ind.ring.mode == "spin" and "30%" in ind.text()
    v = tasks.show_viewer(w)
    qtbot.wait(50)
    assert v.table.rowCount() >= 1 and v.table.item(0, 1).text() == "Scanning 10 files"
    assert isinstance(v.table.cellWidget(0, 6), QPushButton)            # Cancel
    t.finish()
    qtbot.wait(150)
    assert ind.ring.mode == "done" and "Done" in ind.text()
    qtbot.waitUntil(lambda: ind.ring.finished_drawing(), timeout=2000)
    qtbot.waitUntil(lambda: ind.isHidden(), timeout=4000)
    assert v.table.item(0, 2).text() == "done" and "finished" in v.summary.text()
    t2 = tasks.start("Export")
    t2.cancel_fn = lambda: None
    t2.cancel()
    t2.finish()
    qtbot.wait(150)
    assert v.table.item(0, 2).text() == "cancelled"
    v.clear_btn.click()
    assert v.table.rowCount() == 0
    assert w.a.bg_tasks.icon() is not None and not w.a.bg_tasks.icon().isNull()
    v.close()


def test_settings_reset_icon(shell, qtbot):  # noqa: F811
    from AIO_QC.widgets.settings_dialog import SettingsDialog
    w = shell.window("image")
    d = SettingsDialog(w, w.settings_sections(), "Settings")
    r = next(iter(d.rows.values()))
    assert r.reset.text() == "" and not r.reset.icon().isNull()
    d.close()


def test_toast_shrinks_back_after_details(shell, qtbot):  # noqa: F811
    w = shell.window("image")
    w.resize(1400, 900)
    t = w.toast.show_message("Short note", "warn", details="long line " * 40, sticky=True)
    qtbot.wait(350)
    w0 = t.width()
    t.toggle_details()
    qtbot.wait(600)
    assert t.width() > w0 + 100
    t.toggle_details()
    qtbot.wait(600)
    assert abs(t.width() - w0) <= 2
    t.dismiss()
