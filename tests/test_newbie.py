# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Easy to start with: start pages, samples, the tour, the health check, Simple menus, explained buttons, undo."""
import json

from PyQt6.QtWidgets import QMessageBox

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401
from .test_modes import shell3  # noqa: F401


def test_sample_dataset_has_its_planted_problems(tmp_path, monkeypatch):
    from AIO_QC.core import sample_data as SD
    monkeypatch.setattr(SD, "samples_root", lambda: tmp_path)
    root = SD.sample_dataset()
    names = {p.stem for p in root.glob("*.json")}
    assert len(list(root.glob("*.jpg"))) == SD.N_IMAGES and "street_0019" not in names          # image only
    labels = {s["label"] for p in root.glob("*.json") for s in json.loads(p.read_text())["shapes"]}
    assert {"car", "Car", "person", "plate"} <= labels
    assert json.loads((root / "street_0016.json").read_text())["shapes"] == []
    assert (root / "street_0021.jpg").read_bytes() == (root / "street_0022.jpg").read_bytes()
    assert SD.sample_dataset() == root                                                       # made once
    t = SD.sample_table()
    assert t.read_text().count("\n") == 241


def test_welcome_page_until_a_dataset_opens(shell, qtbot, tmp_path):  # noqa: F811
    w = shell.window("image")
    page = w.start_page
    assert not page.isHidden() and page.geometry().top() >= w.toolbar.geometry().bottom()
    assert page.title.text() == "Welcome to AIO QC" and len(page._cards) == 3
    _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    assert page.isHidden()
    page.refresh()
    assert page.recent_box.count() >= 1                                     # the dataset is now a recent item


def test_health_check_finds_the_planted_problems(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core import sample_data as SD
    from AIO_QC.widgets import coach
    monkeypatch.setattr(SD, "samples_root", lambda: tmp_path)
    coach.mark_done("image")
    w = shell.window("image")
    w.open_sample_dataset()
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    assert w.toast.find("sample") is not None                                # it says what to do next
    d = w.open_health_check()
    qtbot.waitUntil(lambda: d.inbox_btn.isEnabled(), timeout=60000)
    r = d.results
    assert r["pairs"] >= 1 and r["ambiguity"] >= 1 and r["empty"] >= 1 and r["image_dups"] >= 2
    assert r["duplicates"] >= 1 and r["validation"] >= 1
    assert "found something" in d.summary.text()
    assert w.open_dialogs.get("pairs") is not None and not w.open_dialogs["pairs"].isVisible()   # ready to open
    d.close()


def test_tour_steps_and_remembers(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets import coach
    coach.mark_done("image", False)
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: getattr(w, "_coach", None) is not None, timeout=3000)     # first dataset: the tour
    t = w._coach
    assert t.i == 0 and not t.hole.isNull()                                 # the Open button is lit
    for _ in range(len(t.steps) - 1):
        t.next()
    assert t.bubble.next.text() == "Done"
    t.next()
    assert coach.done("image") and w._coach is None
    assert coach.run(w, w.tour_steps(), "image", force=False) is None      # not again by itself


def test_simple_menus_hide_but_keep_shortcuts(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets import ui_level
    from AIO_QC.widgets.command_palette import collect_actions
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    act = w.qc_actions["model_check"]
    review = w.menus.qc
    ui_level.set_simple(True)
    shown = [a for m in review.findChildren(type(review)) for a in m.actions()]
    assert act not in shown and act.isVisible() and act.isEnabled()          # out of the menu, shortcut still live
    assert any(a is act for _p, a in collect_actions(w.menuBar()))          # Ctrl+K still finds it
    page = next(w.tabs.widget(i) for i in range(w.tabs.count())
                if any(b.defaultAction() is w.qc_actions["ocr_check"] for b in w.tabs.widget(i).groups[0] +
                       sum(w.tabs.widget(i).groups[1:], [])))
    assert all(b.defaultAction() is not w.qc_actions["ocr_check"] for b in page.buttons())
    ui_level.set_simple(False)
    shown = [a for m in review.findChildren(type(review)) for a in m.actions()]
    assert act in shown
    assert any(b.defaultAction() is w.qc_actions["ocr_check"] for b in page.buttons())


def test_disabled_buttons_say_why(shell, qtbot):  # noqa: F811
    w = shell.window("image")
    tip = w.qc_actions["validation"].toolTip()
    assert "Not available yet: open a dataset first" in tip and "Ctrl+Shift+V" in tip


def test_trash_can_be_undone_from_the_notification(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    keys = [s.key for s in w.session.samples[:2]]
    paths = [w.session.by_key[k].ann_path for k in keys]
    w.session.trash_samples(keys)
    assert not any(p.exists() for p in paths) and len(w.session.samples) == 10
    t = w.toast.find("undo-trash")
    assert t is not None
    undo = next(b for b in t.buttons if b.text() == "Undo")
    undo.click()
    assert all(p.exists() for p in paths) and len(w.session.samples) == 12


def test_bulk_change_offers_undo(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    from AIO_QC.core.shape_ops import fingerprint
    k = w.session.samples[0].key
    ann = w.session.adapter.load(w.session.by_key[k])
    before = w.session.by_key[k].ann_path.read_bytes()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))
    st = w.session.apply_to_shapes(w, {k: [(0, fingerprint(ann.shapes[0]))]}, "delete", what="dimarea")
    assert st["files"] == 1
    t = w.toast.find("undo-bulk")
    assert t is not None and "Changed 1 file" in t.title.text()
    next(b for b in t.buttons if b.text() == "Undo").click()
    assert w.session.by_key[k].ann_path.read_bytes() == before


def test_panel_help_only_where_the_name_is_not_enough(shell, shell3):  # noqa: F811
    w = shell.window("image")
    assert getattr(w.files_title, "help_btn", None) is None              # Files / Annotations / Labels: no "?"
    assert getattr(shell3.window("data").tables_title, "help_btn", None) is not None


def test_studio_start_page_sample_and_tour(shell3, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core import sample_data as SD
    monkeypatch.setattr(SD, "samples_root", lambda: tmp_path)
    sh = shell3
    sh.show("data")
    st = sh.studio
    assert st.stack.currentIndex() == 0 and st.welcome.title.text() == "Clean and explore a table"
    t = st.start_tour(force=True, empty_ok=True)                    # from the start page: opens the sample first
    qtbot.waitUntil(lambda: st.df is not None, timeout=20000)
    assert st.stack.currentIndex() == 1 and t is not None
    t.finish()
    assert "Not available yet" not in st.a.export.toolTip()


def test_frame_extractor_start_page(shell3, qtbot):  # noqa: F811
    sh = shell3
    v = sh.show("video")
    assert v.start_page is not None and not v.start_page.isHidden()
    assert "Not available yet: load a video first" in v.actions_.save.toolTip()
    v.load_video("/tmp/aioqc_test_media_v1/a.mp4")
    qtbot.waitUntil(lambda: v.video_path is not None, timeout=20000)
    assert v.start_page.isHidden()


def test_help_menu_has_one_of_each_in_every_mode(shell3, qtbot):  # noqa: F811
    """Quick Tour · User Guide · Keyboard Shortcuts — the same three, first, in every mode; no slides, no second
    shortcut list, no separate tool map (it is in the guide)."""
    for mode in ("image", "video", "data"):
        w = shell3.show(mode)
        help_menu = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "Help")
        names = [a.text().replace("&", "") for a in getattr(help_menu, "_aio_all_actions", help_menu.actions())
                 if not a.isSeparator()]
        assert names[:3] == ["Quick Tour", "User Guide", "Keyboard Shortcuts"], (mode, names)
        assert not {"Get Started", "Which tool for what?", "Key Map"} & set(names)
        keys = next(a for a in help_menu.actions() if a.text() == "Keyboard Shortcuts")
        assert "?" in [k.toString() for k in keys.shortcuts()]


def test_which_tool_for_what_is_a_section_of_the_guide(shell3, qtbot):  # noqa: F811
    from AIO_QC.help_guide import guide_html
    from AIO_QC.app import QC_TOOLS
    html = guide_html("x")
    assert "name='toolmap'" in html and all(t.title in html for t in QC_TOOLS if t.group)
    from AIO_QC.widgets.help_center_bridge import _actions, open_help
    shell3.show("image")
    _actions()["toolmap"]()
    hc = open_help(shell3.window("image"))
    assert hc._hist[hc._pos] == "aio-image-qc"
    hc.close()


def test_polygon_drawn_and_saved_with_picked_label(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from AIO_QC.widgets import label_picker
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    monkeypatch.setattr(label_picker.LabelPicker, "exec_at", lambda self, at: self.names[0])
    c = w.canvas
    n = len(w.session.load(w.current_key).shapes)
    w.a.new_polygon.trigger()
    assert c._drawing and c._draw_kind == "polygon"
    from AIO_QC.widgets.screenshot import visible_image_rect
    c.apply_default_zoom()
    a = visible_image_rect(c)
    for fx, fy in ((0.2, 0.2), (0.7, 0.25), (0.6, 0.8)):
        pt = QPoint(a.left() + int(a.width() * fx), a.top() + int(a.height() * fy))
        QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    QTest.keyClick(c, Qt.Key.Key_Return)
    shapes = w.session.load(w.current_key).shapes
    assert len(shapes) == n + 1 and shapes[-1]["shape_type"] == "polygon" and len(shapes[-1]["points"]) == 3
    assert w.a.new_polygon.shortcut().toString() == "Ctrl+N"


def test_delete_asks_first(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    c = w.canvas
    n = len(w.session.load(w.current_key).shapes)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: int(QMessageBox.StandardButton.Cancel))
    c.select_shape(c.shape_groups[0].index)
    c.shapeDeleteRequested.emit(c.selected_group.index)
    assert len(w.session.load(w.current_key).shapes) == n                  # cancelled: nothing deleted
    w.settings.setValue("edit/confirm_delete", False)
    c.select_shape(c.shape_groups[0].index)
    c.shapeDeleteRequested.emit(c.selected_group.index)
    assert len(w.session.load(w.current_key).shapes) == n - 1               # "don't ask again"
    w.settings.setValue("edit/confirm_delete", True)


def test_welcome_has_no_status_bar(shell):  # noqa: F811
    w = shell.window("image")
    assert w.session is None and w.statusBar().isHidden()


def test_drawing_and_view_keys(shell, qtbot, tmp_path):  # noqa: F811
    """Ctrl+R new box, Ctrl+N new polygon, Ctrl+E rename / merge, Ctrl+F fit — and nothing else on those keys."""
    from PyQt6.QtGui import QAction
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    keys = lambda a: [k.toString() for k in a.shortcuts()]      # noqa: E731
    assert keys(w.a.new_box) == ["Ctrl+R"]
    assert keys(w.a.rename) == ["Ctrl+E"] and keys(w.a.fit) == ["Ctrl+F"]
    assert keys(w.a.search) == ["Ctrl+P"] and keys(w.qc_actions["fix_editor"]) == ["Ctrl+Alt+O"]
    owners = {}
    for a in w.findChildren(QAction):
        for k in keys(a):
            owners.setdefault(k, set()).add(id(a))
    for k in ("Ctrl+R", "Ctrl+N", "Ctrl+E", "Ctrl+F", "Ctrl+P", "Ctrl+Alt+O"):
        assert len(owners.get(k, ())) == 1, k
    assert "B" not in owners and "Ctrl+0" not in owners                    # left free


def test_keyboard_shortcuts_has_a_close_button(shell, qtbot):  # noqa: F811
    w = shell.window("image")
    w.toggle_keymap()
    km = w._keymap
    qtbot.waitUntil(lambda: km.isVisible() and km.t > 0.99, timeout=2000)
    assert km.close_btn.isVisible() and "Esc" in km.close_btn.toolTip()
    km.close_btn.click()
    qtbot.waitUntil(lambda: km.isHidden(), timeout=2000)


def test_polygon_drawing_keys_snap_and_cancel(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    """Esc cancels, Ctrl+Z takes a corner back (not the file's undo), Ctrl+Shift+Z puts it back, the first corner
    snaps and lights up, and a click on it closes the polygon."""
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from AIO_QC.widgets import label_picker
    from AIO_QC.widgets.screenshot import visible_image_rect
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    monkeypatch.setattr(label_picker.LabelPicker, "exec_at", lambda self, at: self.names[0])
    c = w.canvas
    c.apply_default_zoom()
    a = visible_image_rect(c)
    at = lambda fx, fy: QPoint(a.left() + int(a.width() * fx), a.top() + int(a.height() * fy))   # noqa: E731
    n = len(w.session.load(w.current_key).shapes)
    w.file_dock.setFocus()                                        # the focus elsewhere: drawing takes it
    w.a.new_polygon.trigger()
    assert w.a.new_polygon.text() == "Create new polygon" and w.a.new_box.text() == "Create new bbox"
    assert c.hasFocus() and c._draw_hint is not None and not c._draw_hint.isHidden()
    for p in (at(.2, .2), at(.7, .25), at(.6, .8)):
        QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
    QTest.keyClick(c, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(c._poly_pts) == 2 and c._drawing                   # one corner back, still drawing
    QTest.keyClick(c, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert len(c._poly_pts) == 3
    assert len(w.session.load(w.current_key).shapes) == n         # the file's undo was not touched
    QTest.mouseMove(c.viewport(), at(.2, .2) + QPoint(4, 3))      # near the first corner: it snaps
    c._poly_mouse = at(.2, .2) + QPoint(4, 3)
    c._poly_follow(Qt.KeyboardModifier.NoModifier)
    assert c._poly_snap and c._poly_cursor == c._poly_pts[0] and "close" in c._draw_hint.text()
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(.2, .2) + QPoint(4, 3))
    assert not c._drawing and c._draw_hint.isHidden()
    shapes = w.session.load(w.current_key).shapes
    assert len(shapes) == n + 1 and len(shapes[-1]["points"]) == 3
    # Esc cancels, even with the window's Esc (clear selection) on the same key
    w.a.new_polygon.trigger()
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(.3, .3))
    QTest.keyClick(c, Qt.Key.Key_Escape)
    assert not c._drawing and c._poly_item is None
    # too few corners: Enter keeps the drawing going
    w.a.new_polygon.trigger()
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(.3, .3))
    QTest.keyClick(c, Qt.Key.Key_Return)
    assert c._drawing and len(c._poly_pts) == 1
    c.cancel_drawing()


def test_polygon_shift_keeps_45_degrees(shell, qtbot, tmp_path):  # noqa: F811
    import math
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from AIO_QC.widgets.screenshot import visible_image_rect
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    c = w.canvas
    c.apply_default_zoom()
    a = visible_image_rect(c)
    w.a.new_polygon.trigger()
    p0 = QPoint(a.left() + a.width() // 4, a.top() + a.height() // 4)
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
    q, snap = c._poly_target(p0 + QPoint(60, 7), Qt.KeyboardModifier.ShiftModifier)
    o = c._poly_pts[0]
    assert not snap and abs(q.y() - o.y()) < 1e-6                 # a nearly flat edge becomes flat
    q, _ = c._poly_target(p0 + QPoint(50, 46), Qt.KeyboardModifier.ShiftModifier)
    assert abs(math.degrees(math.atan2(q.y() - o.y(), q.x() - o.x())) - 45) < 1e-6
    c.cancel_drawing()
