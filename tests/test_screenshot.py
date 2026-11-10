# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Screenshots of the canvas (E / Shift+E / right-click / toolbar) and File list ▸ Copy file name."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMenu

from AIO_QC.widgets.screenshot import capture_view, visible_image_rect

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _win(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds", n=4)
    (root / "sub").mkdir()
    import shutil
    for ext in (".json", ".jpg"):
        shutil.copy(root / f"cam_frame_0001{ext}", root / "sub" / f"cam_frame_0001{ext}")
    w = _open(shell, qtbot, root)
    qtbot.waitUntil(lambda: w.current_key is not None and bool(w.canvas.shape_groups), timeout=10000)
    w.resize(1100, 800)
    w.settings.setValue("screenshot/dir", str(tmp_path / "shots"))
    w.settings.setValue("screenshot/format", "png")
    qtbot.wait(50)
    return w


def _px(img: QImage, x, y):
    c = img.pixelColor(x, y)
    return c.red(), c.green(), c.blue()


def test_capture_is_the_view_cropped_to_the_image(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    c = w.canvas
    c.apply_default_zoom()
    img = capture_view(c)
    area = visible_image_rect(c)
    dpr = c.viewport().devicePixelRatioF()
    assert (img.width(), img.height()) == (round(area.width() * dpr), round(area.height() * dpr))
    assert area.width() < c.viewport().width()                       # 320 × 240 at 100 %: no grey bands saved
    # zoomed in: only the part on screen, at screen resolution
    c.animate_to(4.0, center=QPointF(100, 80), duration=0)
    img = capture_view(c)
    assert img.width() == round(c.viewport().width() * dpr) and img.height() == round(c.viewport().height() * dpr)
    # hover, handles and floating widgets are not in it; the hover comes back afterwards
    w.a.select.trigger()
    c.select_shape(0)
    g = c.shape_groups[1]
    c._set_hover(g)
    capture_view(c)
    assert c.hover_group is g and all(h.isVisible() for h in c._handles)


def test_screenshot_includes_brightness_and_sketch(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    c = w.canvas
    c.apply_default_zoom()
    plain = capture_view(c)
    w.disp["brightness"].setValue(80)
    qtbot.waitUntil(lambda: c.display["brightness"] == 80, timeout=2000)
    bright = capture_view(c)
    x, y = 5, 5
    assert sum(_px(bright, x, y)) > sum(_px(plain, x, y)) + 60
    w._reset_display()
    # sketch: the drawing is in the screenshot, the sketch bar is not
    w.sketch.enter()
    assert w.sketch.active
    before = capture_view(c)
    el = {"t": "rect", "c": "#ff00ff", "w": 12, "pts": [[20, 20], [120, 100]], "fill": True}
    w.sketch.layer.commit(el)
    w.sketch._add(el)
    after = capture_view(c)
    k = after.width() / 320
    assert _px(after, int(20 * k), int(60 * k)) == (255, 0, 255) != _px(before, int(20 * k), int(60 * k))  # border
    assert _px(after, int(60 * k), int(60 * k)) != _px(before, int(60 * k), int(60 * k))                    # fill
    path = w.shots.save()                                            # E works in sketch mode too
    assert path is not None and path.is_file()
    w.sketch.leave()


def test_e_saves_per_file_without_overwriting(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    key = w.current_key
    w.canvas.setFocus()
    QTest.keyClick(w.canvas, Qt.Key.Key_E)
    shots = tmp_path / "shots" / "ds"
    first = shots / f"{Path(key).stem}.png"
    qtbot.waitUntil(first.is_file, timeout=3000)
    w.a.screenshot.trigger()
    assert (shots / f"{Path(key).stem}_2.png").is_file()           # never overwritten
    assert QImage(str(first)).width() > 0
    # files in sub-folders keep their folder (same names never collide)
    sub = next(k for k in w.session.by_key if Path(k).parent.name == "sub")
    w.open_sample(sub)
    qtbot.waitUntil(lambda: w.current_key == sub, timeout=3000)
    p = w.shots.save()
    assert p == shots / "sub" / "cam_frame_0001.png"
    # JPEG
    w.settings.setValue("screenshot/format", "jpg")
    p = w.shots.save()
    assert p.suffix == ".jpg" and QImage(str(p)).width() > 0
    # copy
    QApplication.clipboard().clear()
    assert w.shots.copy() and not QApplication.clipboard().image().isNull()
    # toast with "Open folder"
    t = w.toast.find("screenshot")
    assert t is not None


def test_right_click_menu_only_on_the_image(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    c = w.canvas
    c.apply_default_zoom()
    shown = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a: shown.append([x.text() for x in self.actions()]))
    area = visible_image_rect(c)
    QTest.mouseClick(c.viewport(), Qt.MouseButton.RightButton, pos=area.center())
    from PyQt6.QtGui import QContextMenuEvent
    QApplication.sendEvent(c.viewport(), QContextMenuEvent(QContextMenuEvent.Reason.Mouse, area.center()))
    assert shown and shown[-1][0] == "Save Screenshot" and "Copy Screenshot" in shown[-1]
    n = len(shown)
    outside = QPoint(area.right() + 20, area.center().y())
    assert outside.x() < c.viewport().width()
    QApplication.sendEvent(c.viewport(), QContextMenuEvent(QContextMenuEvent.Reason.Mouse, outside))
    assert len(shown) == n                                           # outside the image: no menu
    w.roi.enter() if hasattr(w.roi, "enter") else w.roi.toggle()
    if w.roi.active:                                                 # ROI mode keeps right-click (delete a box)
        QApplication.sendEvent(c.viewport(), QContextMenuEvent(QContextMenuEvent.Reason.Mouse, area.center()))
        assert len(shown) == n
        w.roi.leave()
    # toolbar and File menu
    tb = [getattr(w.toolbar.widgetForAction(a), "defaultAction", lambda: a)() for a in w.toolbar.actions()]
    assert w.a.screenshot in tb and w.a.screenshot.shortcut().toString() == "E"
    assert w.a.screenshot_as.shortcut().toString() == "Shift+E"


def test_copy_file_names(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    keys = [s.key for s in w.session.samples[:2]]
    assert w.copy_file_names(keys[:1]) == Path(keys[0]).name
    assert QApplication.clipboard().text() == Path(keys[0]).name
    assert w.copy_file_names(keys).split("\n") == [Path(k).name for k in keys]
    assert w.copy_file_names(keys, full=True).split("\n") == keys
