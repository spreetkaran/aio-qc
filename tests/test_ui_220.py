# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""2.2.0 UI: Paint-style status bar, zoom slider, toasts that expand in place, QC panel tabs, label colours."""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt6.QtCore import QPointF

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_version_and_name():
    import AIO_QC
    assert AIO_QC.__version__ == "2.2.0" and AIO_QC.__appname__ == "AIO QC App"


def test_status_bar_and_zoom_slider(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    sb = w.statusBar()
    qtbot.waitUntil(lambda: sb.image.text.text() == "320 × 240px", timeout=5000)
    w.canvas.report_cursor(w.canvas.mapFromScene(QPointF(120.4, 77.9)))
    assert sb.pointer.text.text().endswith("px") and "," in sb.pointer.text.text()
    w.canvas.report_cursor(None)
    assert sb.pointer.text.text() == ""
    w.canvas.select_shape(0)
    assert sb.selection.text.text() == "100 × 80px"
    w.canvas.clearSelection()
    assert sb.selection.text.text() == ""
    sb.showMessage("hello there", 2000)
    assert sb.message.text() == "hello there" and sb.pointer.isVisible()     # fields stay while a message shows
    # toolbar order: zoom box, fit, 1:1, zoom out, slider, zoom in
    tb = w.toolbar
    items = [tb.widgetForAction(a) for a in tb.actions()]
    i = items.index(w.zoom_spin)
    act = lambda k: getattr(items[i + k], "defaultAction", lambda: None)()
    assert act(1) is w.a.fit and act(2) is w.a.actual and act(3) is w.a.zoom_out
    assert items[i + 4] is w.zoom_slider and act(5) is w.a.zoom_in
    # the slider zooms live and follows the canvas
    w.zoom_slider.slider.setValue(w.zoom_slider._to_pos(2.0))
    assert abs(w.canvas.current_zoom - 2.0) < 0.02
    w.canvas.zoom_to(0.5)
    qtbot.waitUntil(lambda: abs(w.zoom_slider._to_zoom(w.zoom_slider.slider.value()) - 0.5) < 0.02, timeout=3000)


def test_toast_details_expand_in_place(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtWidgets import QDialog
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w._external_log = [("10:00:00", "changed", k) for k in w.session.by_key] * 4
    t = w.toast.show_message("<b>Updated</b>", details=w._external_log_text, key="external")
    qtbot.wait(300)
    h0, w0, y0 = t.height(), t.width(), t.y()
    t.toggle_details()
    qtbot.waitUntil(lambda: t._expand_anim is None, timeout=3000)
    assert t.height() > h0 + 80 and t.width() >= w0 and t.y() < y0           # grew upwards, anchored at the bottom
    assert "cam_frame_0000" in t.panel.toPlainText()
    assert not t._should_run()                                               # the countdown waits while open
    assert not [x for x in qtbot_dialogs() if isinstance(x, QDialog)]        # no extra window
    t.toggle_details()
    qtbot.waitUntil(lambda: t._expand_anim is None, timeout=3000)
    assert abs(t.height() - h0) <= 8 and not t.panel.isVisible()
    t.dismiss()
    qtbot.wait(300)


def qtbot_dialogs():
    from PyQt6.QtWidgets import QApplication, QDialog
    return [x for x in QApplication.topLevelWidgets() if isinstance(x, QDialog) and x.isVisible()]


def test_ribbon_tab_not_lit_when_closed(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    r = w.tabs
    r.set_pinned(False)
    r.setCurrentIndex(1)
    r.open_drop()
    r.bar.grab()
    r.close_drop(animate=False)
    assert not r._open
    r.bar.grab()                                    # paints with no selected / hovered tab (no exception)


def test_label_colours_distinct_and_saved(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.utils import colors as C
    from AIO_QC.utils.colors import _srgb_to_lab
    from PyQt6.QtGui import QColor
    pal = C.palette()
    assert len(set(pal)) == len(pal) == C.MAX_COLORS
    lab = _srgb_to_lab(np.array([QColor(h).getRgbF()[:3] for h in pal[:12]], float))
    d = np.linalg.norm(lab[:, None] - lab[None], axis=2) + np.eye(12) * 999
    assert d.min() > 30                                                   # the first 12 are clearly apart
    labels = {}
    for n in ("a", "b", "c"):
        C.assign(n, labels)
    assert [labels[n].name() for n in "abc"] == list(pal[:3])
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: os.path.isfile(os.path.join(w.session.root, ".aio_qc", C.FILE)), timeout=10000)
    saved = json.load(open(os.path.join(w.session.root, ".aio_qc", C.FILE)))["colors"]
    assert {"car", "plate", "wheel"} <= set(saved)
    assert len(set(saved.values())) == len(saved)                         # every label its own colour
    keep = dict(saved)
    w._close_session() if hasattr(w, "_close_session") else None
    w2 = _open(shell, qtbot, w.session.root if w.session else tmp_path / "ds")
    assert {k: v.name() for k, v in w2.label_colors.items() if k in keep} == keep     # same colours next time


def test_outline_width_constant_on_zoom(shell, qtbot, tmp_path):  # noqa: F811
    """Shape outlines are cosmetic: the same screen width at every zoom (never covering the pixels being checked)."""
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: bool(w.canvas.shape_groups), timeout=10000)
    for g in w.canvas.shape_groups:
        pen = g.shape_item.pen()
        assert pen.isCosmetic() and pen.widthF() <= 4.0
    w.canvas.animate_to(8.0, duration=0)
    assert all(g.shape_item.pen().isCosmetic() for g in w.canvas.shape_groups)


def test_point_dots_constant_on_zoom(shell, qtbot, tmp_path):  # noqa: F811
    """Point shapes are dots of a fixed screen size, like outlines and tags."""
    root = FX.labelme(tmp_path / "ds", n=2)
    f = sorted(root.glob("*.json") if hasattr(root, "glob") else __import__("pathlib").Path(root).glob("*.json"))[0]
    d = json.load(open(f))
    d["shapes"].append({"label": "kp", "points": [[200, 60]], "shape_type": "point", "flags": {}, "group_id": None,
                        "description": ""})
    json.dump(d, open(f, "w"))
    w = _open(shell, qtbot, root)
    from AIO_QC.widgets.canvas import PointsItem
    qtbot.waitUntil(lambda: any(isinstance(g.shape_item, PointsItem) for g in w.canvas.shape_groups), timeout=10000)
    it = next(g.shape_item for g in w.canvas.shape_groups if isinstance(g.shape_item, PointsItem))
    sizes = []
    for z in (1.0, 8.0):
        w.canvas.animate_to(z, duration=0)
        sizes.append(w.canvas.mapFromScene(it.boundingRect()).boundingRect().width())
    assert abs(sizes[0] - sizes[1]) <= 3                         # same on-screen size at 1× and 8×
    w.canvas.viewport().grab()


def test_zoom_box_and_slider_follow_every_step(shell, qtbot, tmp_path):  # noqa: F811
    """N quick clicks on zoom in = N steps, and the zoom box / slider move with the glide (Windows Photos)."""
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    c = w.canvas
    c.animate_to(1.0, duration=0)
    seen = []
    c.zoomChanged.connect(lambda z: seen.append(w.zoom_spin.value.text()))
    for _ in range(3):
        w.a.zoom_in.trigger()
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=5000)
    assert abs(c.current_zoom - 1.3) < 0.005                         # three clicks: 110, 120, 130 %
    assert len(set(seen)) > 3                                         # the box changed during the glide
    assert w.zoom_spin.value.text() == w.zoom_spin.fmt(c.current_zoom)
    assert abs(w.zoom_slider._to_zoom(w.zoom_slider.slider.value()) - c.current_zoom) < 0.02


def test_status_file_type_and_size(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    sb = w.statusBar()
    qtbot.waitUntil(lambda: sb.ftype.text.text() == "JPG", timeout=5000)
    img = w.current_ann.image_path
    assert sb.fsize.text.text().endswith("KB") and str(os.path.getsize(img)) in sb.fsize.toolTip().replace(",", "")
    from AIO_QC.widgets.status_fields import file_type, human_size
    assert file_type("a.png") == "PNG" and file_type("b.JPG") == "JPG" and file_type("c.jpeg") == "JPEG" \
        and file_type("c.tif") == "TIF"
    assert human_size(143155) == "139.8 KB" and human_size(500) == "500 bytes"


def test_hidden_label_selection_from_annotations_dock(shell, qtbot, tmp_path):  # noqa: F811
    """Clicking a greyed (hidden-label) annotation shows that shape while selected; clicking elsewhere / the same
    row again deselects it and hides it again — the list and the canvas never disagree."""
    from PyQt6.QtCore import Qt
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None and bool(w.canvas.shape_groups), timeout=10000)
    w.visible_labels.discard("plate")
    w.load_canvas_shapes()
    w.update_annotations_list()
    c, lst = w.canvas, w.annotations_list
    plate = next(i for i, s in enumerate(w.current_ann.shapes) if s["label"] == "plate")
    assert plate not in c.index_to_group
    row = next(r for r in range(lst.count()) if lst.item(r).data(Qt.ItemDataRole.UserRole) == plate)
    lst.setCurrentRow(row)                                      # = clicking the greyed row
    assert c.selected_group is not None and c.selected_group.index == plate and c.is_peeked(plate)
    c.clearSelection()                                          # = clicking elsewhere on the image
    assert c.selected_group is None and plate not in c.index_to_group
    assert lst.currentRow() == -1 and not lst.selectedItems()
    # a visible shape after a peeked one: the peeked one goes away
    lst.setCurrentRow(row)
    car = next(i for i, s in enumerate(w.current_ann.shapes) if s["label"] == "car")
    c.select_shape(car)
    assert plate not in c.index_to_group and c.selected_group.index == car
    # hiding the label of the selected shape clears the selection everywhere
    w.visible_labels.discard("car")
    w.load_canvas_shapes()
    assert c.selected_group is None and lst.currentRow() == -1
    # a QC tool opening a hidden shape (focus) works too
    c.focus_shape(plate)
    assert c.selected_group is not None and c.selected_group.index == plate


def test_icons_sharp_at_any_scaling():
    """App / mode icons ship every common size; painted icons render at the exact device size."""
    from PyQt6.QtCore import QSize
    from AIO_QC.utils import get_color_icon, newIcon
    pm = newIcon("mode-video").pixmap(QSize(96, 96), 3.0)        # the Frame Extractor icon is a vector
    assert pm.width() == 288 and not pm.toImage().isNull()
    for name in ("app",):
        sizes = {s.width() for s in newIcon(name).availableSizes()}
        assert {16, 20, 24, 32, 40, 48, 64, 128, 256, 512} <= sizes, (name, sizes)
    for dpr in (1.0, 1.25, 1.5, 2.0, 4.0):
        pm = get_color_icon("#ff0000").pixmap(QSize(12, 12), dpr)
        assert pm.width() == round(12 * dpr) and abs(pm.devicePixelRatio() - dpr) < 1e-6
        pm = newIcon("open").pixmap(QSize(16, 16), dpr)
        assert pm.width() == round(16 * dpr)


def test_mode_switch_icons(shell, qtbot):  # noqa: F811
    from AIO_QC.widgets.mode_switch import ModeSwitch
    sw = ModeSwitch("image")
    qtbot.addWidget(sw)
    for b in sw.buttons.values():
        assert b.text().startswith(" ") and not b.icon().isNull()
    assert {s.width() for s in sw.buttons["image"].icon().availableSizes()} >= {16, 32}


def test_zoom_buttons_hold_to_zoom(shell, qtbot, tmp_path):  # noqa: F811
    """Windows Photos: a click = one step (Settings ▸ Viewer, default 15 %); press and hold = that step, a short pause, then a continuous smooth
    zoom until release (which eases out)."""
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from AIO_QC.widgets.zoom_widget import HoldToZoom
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    c = w.canvas
    btn = next(b for b in (w.toolbar.widgetForAction(a) for a in w.toolbar.actions())
               if getattr(b, "defaultAction", lambda: None)() is w.a.zoom_in)
    K = 1.1                                                        # Windows Photos: 100 % -> 110 %
    c.animate_to(1.0, duration=0)
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    assert abs(c.current_zoom - K) < 0.01                          # a click: exactly one step
    for _ in range(2):                                              # quick repeated clicks: one step each
        QTest.mouseClick(btn, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    assert abs(c.current_zoom - 1.3) < 0.005
    c.animate_to(1.0, duration=0)
    import math
    import time as _t
    zs = []
    c.zoomChanged.connect(lambda z: zs.append((_t.perf_counter(), z)))
    QTest.mousePress(btn, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(HoldToZoom.DELAY - 120)
    assert abs(c.current_zoom - K) < 0.01                          # the first step at once, then a pause
    qtbot.wait(900)
    QTest.mouseRelease(btn, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    assert c.current_zoom > K * 1.5                              # then it kept zooming
    z_only = [z for _t0, z in zs]
    assert len(zs) > 30 and all(b >= a - 1e-9 for a, b in zip(z_only, z_only[1:]))   # never backwards
    cont = [(t, z) for t, z in zs if z >= K - 1e-6]             # after the first step's glide landed
    speeds, i = [], 0                                               # log-zoom per second over ≥ 40 ms windows
    for j in range(len(cont)):
        if cont[j][0] - cont[i][0] >= 0.04:
            speeds.append(math.log(cont[j][1] / cont[i][1]) / (cont[j][0] - cont[i][0]))
            i = j
    assert max(speeds) <= HoldToZoom.RATE * 1.2                     # never faster than the steady speed: no jumps
    btn_out = next(b for b in (w.toolbar.widgetForAction(a) for a in w.toolbar.actions())
                   if getattr(b, "defaultAction", lambda: None)() is w.a.zoom_out)
    from AIO_QC.widgets.zoom_widget import photos_step
    z1 = c.current_zoom
    QTest.mouseClick(btn_out, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    assert abs(c.current_zoom - photos_step(z1, -1)) < 0.005


def test_high_quality_zoom_like_photos(shell, qtbot, tmp_path):  # noqa: F811
    """At rest the visible part is re-rendered in high quality (bicubic up, area down) — no blocky pixels when
    zoomed far in, no aliasing when zoomed out; View ▸ Show Pixels When Zoomed In keeps hard pixels instead."""
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    c, item = w.canvas, w.canvas.pixmap_item
    if w.a.pixel_view.isChecked():                   # another test may have toggled every action (saved setting)
        w.a.pixel_view.trigger()
    w.resize(1000, 800)
    for z in (8.0, 0.3):
        c.animate_to(z, duration=0)
        c.viewport().repaint()
        qtbot.waitUntil(lambda: item.hq is not None and abs(item.hq[0] - c.viewportTransform().m11()) < 1e-9,
                        timeout=3000)
        assert c.renderHints() & c.renderHints().__class__.SmoothPixmapTransform
        c.viewport().grab()
    w.a.pixel_view.trigger()
    c.animate_to(8.0, duration=0)
    c.viewport().repaint()
    qtbot.wait(250)
    assert item.hq is None or abs(item.hq[0] - c.viewportTransform().m11()) > 1e-9
    assert not (c.renderHints() & c.renderHints().__class__.SmoothPixmapTransform)
    w.a.pixel_view.trigger()
    assert not w.a.pixel_view.isChecked()


def test_pixel_view_lives_in_settings(shell, qtbot, tmp_path):  # noqa: F811
    """Settings ▸ Viewer holds 'Show pixels when zoomed in' (not the View menu); it drives the canvas."""
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    secs = {s.title: s for s in w.settings_sections()}
    assert "Viewer" in secs
    st = next(x for x in secs["Viewer"].settings if x.key == "image.pixel_view")
    assert w.a.pixel_view not in w.menus.view.actions()
    before = w.a.pixel_view.isChecked()
    st.set(not before)
    assert w.canvas.pixmap_item.pixel_view is (not before)
    st.set(before)
    assert w.canvas.pixmap_item.pixel_view is before


def test_title_without_version_and_about_everywhere(shell, qtbot):  # noqa: F811
    """LabelMe-style About in every mode: icon left; name, purpose, version, author right; OK bottom right."""
    import AIO_QC
    from AIO_QC.utils import appIcon
    from AIO_QC.widgets.about import AboutDialog
    w = shell.window("image")
    assert w.windowTitle() in ("Image QC — AIO QC App",) or w.windowTitle().startswith("Image QC — ")
    assert w.windowTitle().endswith(" — AIO QC App") and AIO_QC.__version__ not in w.windowTitle()
    assert w.a.about in w.menus.help.actions()
    d = AboutDialog(w, "Image QC", "Quality control for annotated image datasets", appIcon())
    qtbot.addWidget(d)
    assert d.title.text() == "Image QC" and d.title.font().bold()
    assert d.version.text() == f"Version: {AIO_QC.__version__}" and d.author.text() == "Author: Preetkaran Singh"
    assert d.windowTitle() == "About Image QC" and d.icon.isz == 96
    d.grab()
    from .conftest import MESSAGES
    w.a.about.trigger()
    assert ("dialog", "AboutDialog") in MESSAGES


def test_window_titles_follow_one_pattern(shell, qtbot, tmp_path):  # noqa: F811
    """'<mode> — <what is open> — AIO QC App' in every mode."""
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    assert w.windowTitle() == "Image QC — ds — AIO QC App"
    w._close_session()
    assert w.windowTitle() == "Image QC — AIO QC App"


def test_zoom_levels_like_windows_photos(shell, qtbot, tmp_path):  # noqa: F811
    """10-point steps up to 200 %, 25-point steps up to 600 %, 50-point steps above; clicks snap to that grid."""
    from AIO_QC.widgets.zoom_widget import photos_step as step
    seq, z = [], 1.0
    for _ in range(18):
        z = step(z, 1)
        seq.append(round(z * 100))
    assert seq[:10] == [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    assert seq[10:18] == [225, 250, 275, 300, 325, 350, 375, 400]
    assert round(step(6.0, 1) * 100) == 650 and round(step(6.5, -1) * 100) == 600
    assert round(step(2.25, -1) * 100) == 200 and round(step(2.0, -1) * 100) == 190
    assert round(step(1.37, 1) * 100) == 140 and round(step(1.37, -1) * 100) == 130
    assert step(10.0, 1) == 10.0 and step(0.01, -1) == 0.01
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    c = w.canvas
    c.animate_to(1.9, duration=0)
    for _ in range(2):
        w.a.zoom_in.trigger()
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    assert abs(c.current_zoom - 2.25) < 0.005                       # 190 -> 200 -> 225 %
    secs = {s.title: s for s in w.settings_sections()}
    assert not any(x.key == "image.zoom_step" for x in secs["Viewer"].settings)
