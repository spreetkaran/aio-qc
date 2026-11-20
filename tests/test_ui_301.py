# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""3.0 polish: one window for the modes with its own title bar, steady read-outs and status bar, floating hint,
adaptive QC drawer, fit / 1:1 state, themed tooltips, flagged copy folder named after the dataset."""
from PyQt6.QtWidgets import QToolButton

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_one_window_for_all_modes(shell, qtbot):  # noqa: F811
    from AIO_QC.widgets.window_frame import FrameHost
    w = shell.window("image")
    host = shell.host
    assert isinstance(host, FrameHost) and w.window() is host and host.isVisible()
    assert not host.custom or w._window_controls.close_btn.isVisible()     # minimise / maximise / close
    shell.switch("data")
    assert shell.studio.window() is host and host.current() is shell.studio
    assert shell.studio.menuBar().cornerWidget(__import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.Corner.TopLeftCorner)\
        .objectName() == "brandCorner"                                  # the same mark in every mode
    qtbot.waitUntil(lambda: host._overlay is None, timeout=3000)       # the cross-fade ends
    shell.switch("image")
    assert host.current() is w and w.mode_switch.current == "image"


def test_hud_and_hint_float_over_canvas(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None)
    hud = w.hud
    assert hud.file.parent() is w.canvas and hud.pos.parent() is w.canvas      # not the scrolling viewport
    before = hud.file.pos()
    w.canvas.scrollContentsBy(40, 30)
    assert hud.file.pos() == before
    size = w.canvas.size()
    w.hint_banner.show_hint("Start here: Something", "a long explanation " * 20, [("Run", lambda: None)])
    assert w.canvas.size() == size and w.hint_banner.parent() is w.canvas
    qtbot.waitUntil(lambda: w.hint_banner._anim is None, timeout=2000)          # glided into place
    assert hud.file.y() >= w.hint_banner.geometry().bottom()
    assert "Start here" in w.hint_banner.text.text()
    assert w.minimumSizeHint().width() < 1000
    w.hint_banner.clear()


def test_fit_and_actual_light_up(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None)
    w.canvas.actual_size()
    qtbot.waitUntil(lambda: w.a.actual.isChecked(), timeout=3000)
    assert w._chrome_btns["actual"].isChecked()
    w.canvas.zoom_to(2.37)
    qtbot.waitUntil(lambda: not w.a.actual.isChecked(), timeout=3000)
    assert not w._chrome_btns["actual"].isChecked()


def test_status_bar_slots_and_icons(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None)
    width = w.mode_label.width()
    w.a.select.trigger()
    assert w.mode_label.text() == "Select" and w.mode_label.width() == width
    bc = w.dataset_label
    pw, pill = bc.path.width(), bc.pill.x()
    bc.set_location("train", "a_very_long_file_name_" * 6)
    assert bc.path.width() == pw and bc.pill.x() == pill and "…" in bc.path.text()


def test_ribbon_never_cuts_names(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtGui import QFontMetrics
    from AIO_QC.widgets.ribbon_page import RibbonPage
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    for i in range(w.tabs.count()):
        page = w.tabs.widget(i)
        if not isinstance(page, RibbonPage):
            continue
        for width in (1700, 900, 500):
            page.resize(width, page.height())
            page.relayout()
            for b in page.buttons():
                fm = QFontMetrics(b.font())
                lines = b.lines() if b.large else [b.label]
                assert all("…" not in x for x in lines)
                assert max(fm.horizontalAdvance(x) for x in lines) <= b.sizeHint().width()


def test_tooltips_formatted_but_plain_to_code(qapp):
    from AIO_QC.widgets import design
    design.install_tooltips()
    b = QToolButton()
    b.setToolTip("Mark OK (S) — Mark current file as reviewed OK")
    assert b.toolTip() == "Mark OK (S) — Mark current file as reviewed OK"
    html = design.format_tip(b.toolTip())
    assert html.startswith("<p") and "Mark OK" in html and ">S<" in html
    assert design.format_tip("Fit image to window") == "Fit image to window"


def test_flagged_copy_folder_named_after_dataset(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.utils import app_settings
    app_settings().setValue("flagged/folder", str(tmp_path / "old_flagged"))
    app_settings().setValue("flagged/base", "")
    w = _open(shell, qtbot, FX.labelme(tmp_path / "LPR"))
    from AIO_QC.widgets.flagged_dialog import FlaggedFilesDialog
    d = FlaggedFilesDialog(w)
    assert d.folder.text().endswith("LPR_flagged")
    d.close()


def test_task_indicator_fixed_slot(shell, qtbot):  # noqa: F811
    from AIO_QC.widgets import tasks
    w = shell.window("image")
    ind = w._task_indicator
    t = tasks.start("A job with a rather long name " * 4)
    qtbot.waitUntil(lambda: ind.isVisible(), timeout=2000)
    assert ind.width() == ind.WIDTH
    t.update(0.5, "half way")
    assert ind.width() == ind.WIDTH and ind.text().startswith("A job")
    t.finish()


def test_every_drawer_button_is_clickable(shell, qtbot, tmp_path):  # noqa: F811
    """Nothing (an old layout filler, a divider) may sit over a QC drawer button — at any window width."""
    from AIO_QC.widgets.ribbon_page import RibbonPage
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    host = shell.host
    w.tabs.set_pinned(True)
    for width in (1900, 1300, 1000):
        host.resize(width, 900)
        for i in range(w.tabs.count()):
            page = w.tabs.widget(i)
            if not isinstance(page, RibbonPage):
                continue
            w.tabs.setCurrentIndex(i)
            qtbot.wait(30)
            vp = page.scroll.viewport()
            for b in page.buttons():
                c = b.mapTo(vp, b.rect().center())
                if vp.rect().contains(c):
                    assert vp.childAt(c) is b, (width, i, b.label)
    w.tabs.set_pinned(False)
    hits = []
    stats = w.tabs.widget(0).buttons()[0]
    stats.defaultAction().triggered.connect(lambda *_: hits.append(1))
    w.tabs.open_tab_keyboard(0)
    qtbot.mouseClick(stats, __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton)
    assert hits


def test_tool_windows_get_merged_title_bar(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtCore import Qt
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.qc_actions["validation"].trigger()
    qtbot.waitUntil(lambda: "validation" in w.open_dialogs)
    d = w.open_dialogs["validation"]
    assert d.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert d._window_controls.close_btn.isVisibleTo(d)


def test_rule_forms_and_windows_fit_the_screen(shell, qtbot, tmp_path):  # noqa: F811
    """No window or dialog opens bigger than the screen: long forms scroll instead (Relationship Rules ▸ rule)."""
    from PyQt6.QtGui import QGuiApplication
    from AIO_QC.core import rules as RL
    from AIO_QC.widgets.rules_dialog import RuleForm
    from AIO_QC.widgets import window_frame
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.qc_actions["rules"].trigger()
    qtbot.waitUntil(lambda: "rules" in w.open_dialogs)
    rd = w.open_dialogs["rules"]
    avail = QGuiApplication.primaryScreen().availableGeometry()
    for t in RL.TYPES:
        f = RuleForm(rd, {"type": t}, rd.labels, rd.profile)
        window_frame._try(f)
        f.show()
        qtbot.wait(20)
        g = f.frameGeometry()
        assert g.width() <= avail.width() and g.height() <= avail.height(), (t, g)
        f.close()


def test_frame_extractor_unloads_when_unused_and_reopens(shell, qtbot):  # noqa: F811
    """Settings ▸ General ▸ Unload the Frame Extractor when unused: it closes (state saved) and opens again."""
    from AIO_QC.utils import app_settings
    app_settings().setValue("shell/fx_unload_min", 1)
    shell.switch("video")
    v = shell.video
    assert v is not None and shell.host.current() is v
    shell.switch("image")
    assert shell._unload_timer.isActive()
    shell._unload_video()                                   # (as when the timer fires)
    assert shell.video is None and v not in shell.host.pages
    assert shell.image.isVisible()                          # nothing else closed
    shell.switch("video")
    assert shell.video is not None and shell.host.current() is shell.video
    app_settings().setValue("shell/fx_unload_min", 15)
