# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Keep Previous Zoom and Keep Previous Brightness/Contrast (Image QC), and the fast display renderer."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QImage, QPixmap

from AIO_QC.widgets.canvas import DisplayRenderer, display_lut

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _resize(root, i, w, h):
    """Make frame i a w × h image (and say so in its JSON)."""
    import cv2
    name = f"cam_frame_{i:04d}"
    a = (np.random.default_rng(i).random((h, w, 3)) * 200).astype(np.uint8)
    cv2.imwrite(str(root / f"{name}.jpg"), a)
    d = json.loads((root / f"{name}.json").read_text())
    d["imageWidth"], d["imageHeight"] = w, h
    (root / f"{name}.json").write_text(json.dumps(d))


def _go(w, qtbot, i):
    key = sorted(w.session.by_key)[i]
    w.show_sample(key)
    qtbot.waitUntil(lambda: w.current_key == key, timeout=5000)
    return key


def _settle(c, qtbot):
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)


def _centre(c):
    p = c._view_center()
    pm = c._original
    return p.x() / pm.width(), p.y() / pm.height()


@pytest.fixture
def win(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds", n=6)
    _resize(root, 2, 640, 480)          # same aspect, 2× the size
    _resize(root, 3, 1600, 400)         # another aspect
    _resize(root, 4, 320, 240)
    w = _open(shell, qtbot, root)
    w.resize(1100, 800)
    qtbot.wait(50)
    for act in (w.a.keep_zoom, w.a.keep_display):
        if act.isChecked():
            act.trigger()
    w._reset_display()
    yield w
    for act in (w.a.keep_zoom, w.a.keep_display):
        if act.isChecked():
            act.trigger()
    w._reset_display()


def test_keep_zoom_off_opens_at_home(win, qtbot):
    w, c = win, win.canvas
    _go(w, qtbot, 0)
    c.animate_to(4.0, center=QPointF(60, 50), duration=0)
    _go(w, qtbot, 1)
    assert abs(c.current_zoom - c.home_zoom()) < 1e-6


def test_keep_zoom_same_size_same_spot(win, qtbot):
    w, c = win, win.canvas
    w.a.keep_zoom.trigger()
    assert c.keep_zoom and w.settings.value("view/keep_zoom", type=bool)
    _go(w, qtbot, 0)
    c.animate_to(4.0, center=QPointF(80, 70), duration=0)
    want = c._view_center()
    _go(w, qtbot, 1)
    assert abs(c.current_zoom - 4.0) < 1e-6
    assert (c._view_center() - want).manhattanLength() < 0.5           # the very same pixel
    assert c.zoom_mode == c.CUSTOM_MODE
    # a zoom started just before switching: the glide's target is kept
    c.animate_to(6.0, center=QPointF(200, 150), duration=400)
    _go(w, qtbot, 0)
    assert abs(c.current_zoom - 6.0) < 1e-6 and (c._view_center() - QPointF(200, 150)).manhattanLength() < 0.5


def test_keep_zoom_other_sizes_no_drift(win, qtbot):
    w, c = win, win.canvas
    w.a.keep_zoom.trigger()
    _go(w, qtbot, 0)                                                    # 320 × 240
    c.animate_to(5.0, center=QPointF(300, 220), duration=0)             # near the bottom-right corner
    z0, n0 = c.current_zoom, _centre(c)
    _go(w, qtbot, 2)                                                    # 640 × 480: half the zoom, same spot
    assert abs(c.current_zoom - z0 / 2) < 1e-6
    assert max(abs(a - b) for a, b in zip(_centre(c), n0)) < 0.01
    _go(w, qtbot, 3)                                                    # 1600 × 400: same share of the image
    assert abs(c.current_zoom - z0 * math.sqrt(320 * 240 / (1600 * 400))) < 1e-6
    assert c.clamp_center(c.current_zoom, c._view_center()) == c._view_center()      # never past an edge
    _go(w, qtbot, 4)                                                    # back to 320 × 240: exactly as left
    assert abs(c.current_zoom - z0) < 1e-6 and max(abs(a - b) for a, b in zip(_centre(c), n0)) < 1e-3
    # the user pans on the large image: that is the new spot
    _go(w, qtbot, 2)
    c.animate_to(c.current_zoom * 2, center=QPointF(100, 100), duration=0)
    _go(w, qtbot, 0)
    assert abs(c.current_zoom - z0 * 2) < 1e-6
    assert max(abs(a - b) for a, b in zip(_centre(c), (100 / 640, 100 / 480))) < 0.02


def test_keep_zoom_home_and_fit(win, qtbot):
    w, c = win, win.canvas
    w.a.keep_zoom.trigger()
    _go(w, qtbot, 0)
    c.apply_default_zoom()
    _go(w, qtbot, 2)
    assert abs(c.current_zoom - c.home_zoom()) < 1e-6                  # home stays home
    c.fit_window()
    _settle(c, qtbot)
    _go(w, qtbot, 3)
    assert abs(c.current_zoom - c.compute_fit_scale()) < 1e-6 and c.zoom_mode == c.FIT_MODE     # fit stays fit


def test_reload_keeps_the_view(win, qtbot):
    """The same file re-read after an outside change keeps its view even with Keep Previous Zoom off."""
    w, c = win, win.canvas
    key = _go(w, qtbot, 0)
    c.animate_to(3.0, center=QPointF(100, 90), duration=0)
    pix = QPixmap(c._original)
    pix.detach()
    c.loadPixmap(pix.copy(), same_image=True)
    assert abs(c.current_zoom - 3.0) < 1e-6 and w.current_key == key


def test_keep_zoom_survives_a_broken_image(win, qtbot):
    w, c = win, win.canvas
    w.a.keep_zoom.trigger()
    _go(w, qtbot, 0)
    c.animate_to(4.0, center=QPointF(90, 60), duration=0)
    c.clearAll()                                                        # a file that did not decode
    _go(w, qtbot, 1)
    assert abs(c.current_zoom - 4.0) < 1e-6


def test_new_dataset_forgets_the_zoom(win, qtbot, tmp_path):
    w, c = win, win.canvas
    w.a.keep_zoom.trigger()
    c.animate_to(4.0, center=QPointF(90, 60), duration=0)
    w.open_dataset(FX.labelme(tmp_path / "other", n=3))
    qtbot.waitUntil(lambda: w.session.index_complete and w.current_key is not None, timeout=20000)
    assert abs(c.current_zoom - c.home_zoom()) < 1e-6


def test_brightness_per_image_and_keep(win, qtbot):
    w, c = win, win.canvas
    k0 = _go(w, qtbot, 0)
    w.disp["brightness"].setValue(40)
    w.disp["contrast"].setValue(-20)
    qtbot.waitUntil(lambda: c.display["brightness"] == 40 and c.display["contrast"] == -20, timeout=2000)
    assert w.pixmap_cache.get(k0).cacheKey() != c.pixmap_item.pixmap().cacheKey()      # adjusted
    _go(w, qtbot, 1)                                                    # keep off: opens unadjusted
    assert c.display_is_neutral() and w.disp["brightness"].value() == 0
    assert c.pixmap_item.pixmap().cacheKey() == c._original.cacheKey()
    _go(w, qtbot, 0)                                                    # its own adjustment comes back
    assert w.disp["brightness"].value() == 40 and c.display["contrast"] == -20
    # keep on: carried to the next images
    w.disp_keep.setChecked(True)
    assert w.a.keep_display.isChecked() and w.settings.value("view/keep_display", type=bool)
    _go(w, qtbot, 2)
    assert c.display["brightness"] == 40 and not c.display_is_neutral()
    w.disp_auto.setChecked(True)                                        # a change on this image, recorded for it
    qtbot.waitUntil(lambda: c.display["auto"], timeout=2000)
    _go(w, qtbot, 3)
    assert c.display["auto"]
    w.a.keep_display.trigger()
    assert not w.disp_keep.isChecked()
    _go(w, qtbot, 1)                                                    # never adjusted: unadjusted
    assert c.display_is_neutral()
    _go(w, qtbot, 2)
    assert c.display["auto"] and c.display["brightness"] == 40
    # a slider moved right before switching belongs to the image it was moved on
    w.disp["brightness"].setValue(-30)
    k3 = _go(w, qtbot, 3)
    assert w._display_by_key[sorted(w.session.by_key)[2]]["brightness"] == -30
    assert k3 in w._display_by_key                                      # auto was carried and kept there


def _ref(rgb, d):
    lut = display_lut(d["brightness"], d["contrast"], d["gamma"])
    return lut[rgb]


@pytest.mark.parametrize("fmt", ["rgb", "gray", "alpha", "gray16", "indexed", "odd"])
def test_display_renderer_formats(qapp, fmt):
    rng = np.random.default_rng(1)
    w, h = (321, 77) if fmt == "odd" else (64, 48)
    rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    if fmt == "gray":
        g = np.ascontiguousarray(rgb[:, :, 0])
        img = QImage(g.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        rgb = np.dstack([g, g, g])
    elif fmt == "gray16":
        g16 = (rng.random((h, w)) * 65535).astype(np.uint16)
        img = QImage(g16.data, w, h, 2 * w, QImage.Format.Format_Grayscale16).copy()
        g8 = img.convertToFormat(QImage.Format.Format_RGB32)
        p = g8.constBits()
        p.setsize(g8.sizeInBytes())
        rgb = np.frombuffer(p, np.uint8).reshape(h, w, 4)[:, :, 2::-1].copy()
    elif fmt == "alpha":
        a = (rng.random((h, w, 1)) * 255).astype(np.uint8)
        a[: h // 2] = 255                                               # half opaque, half translucent
        rgba = np.ascontiguousarray(np.dstack([rgb, a]))
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    else:
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        if fmt == "indexed":
            img = img.convertToFormat(QImage.Format.Format_Indexed8)
            back = img.convertToFormat(QImage.Format.Format_RGB888)
            p = back.constBits()
            p.setsize(back.sizeInBytes())
            rgb = np.frombuffer(p, np.uint8).reshape(h, back.bytesPerLine())[:, :3 * w].reshape(h, w, 3).copy()
    pm = QPixmap.fromImage(img)
    d = {"brightness": 25, "contrast": 30, "gamma": 1.4, "auto": False}
    r = DisplayRenderer()
    out = r.render(pm, d).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    p = out.constBits()
    p.setsize(out.sizeInBytes())
    got = np.frombuffer(p, np.uint8).reshape(h, out.bytesPerLine())[:, :4 * w].reshape(h, w, 4)
    if fmt == "alpha":                                                  # transparency kept, colours adjusted
        assert np.array_equal(got[:, :, 3], rgba[:, :, 3])
        opaque = rgba[:, :, 3] == 255
        assert np.abs(got[:, :, :3][opaque].astype(int) - _ref(rgb, d)[opaque]).max() <= 1
    else:
        assert np.abs(got[:, :, :3].astype(int) - _ref(rgb, d)).max() <= 1
    # auto contrast is computed once per image, then only the curve is re-applied
    r.render(pm, dict(d, auto=True))
    cached = r._auto
    r.render(pm, dict(d, auto=True, brightness=-10))
    assert r._auto is cached


@pytest.mark.parametrize("zoom", [4.0, 0.25])
def test_hq_render_keeps_colours(shell, qtbot, zoom):  # noqa: F811
    """The rested high-quality render works on the pixmap's own pixels (no copy) and keeps the channel order."""
    from AIO_QC.widgets.canvas import Canvas
    c = Canvas()
    qtbot.addWidget(c)
    c.resize(500, 400)
    c.show()
    a = np.zeros((400, 600, 3), np.uint8)
    a[:, :300] = (220, 30, 10)                                          # red | blue (RGB)
    a[:, 300:] = (10, 40, 230)
    c.loadPixmap(QPixmap.fromImage(QImage(a.data, 600, 400, 1800, QImage.Format.Format_RGB888).copy()))
    c.animate_to(zoom, center=QPointF(150, 200), duration=0)
    c.pixmap_item.refine()
    hq = c.pixmap_item.hq
    assert hq is not None
    px = hq[2].pixelColor(hq[2].width() // 4 if zoom > 1 else 2, hq[2].height() // 2)
    assert (px.red(), px.green(), px.blue()) == (220, 30, 10)
