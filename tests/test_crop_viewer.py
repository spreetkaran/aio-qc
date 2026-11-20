# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Crop Viewer: an annotation's crop in its own window (Annotations panel ▸ right-click ▸ View crop)."""
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QGuiApplication

from AIO_QC.widgets import crop_viewer as CV

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_outline_and_crop_rect():
    assert CV.shape_outline({"points": [], "shape_type": "polygon"}) is None
    assert CV.shape_outline({"points": [[1, "x"]], "shape_type": "polygon"}) is None
    assert CV.shape_outline({"points": [[float("nan"), 1]], "shape_type": "point"}) is None
    r = CV.shape_outline({"points": [[10, 10], [50, 30]], "shape_type": "rectangle"}).boundingRect()
    assert (r.width(), r.height()) == (40, 20)
    c = CV.shape_outline({"points": [[100, 100], [110, 100]], "shape_type": "circle"}).boundingRect()
    assert abs(c.width() - 20) < 1e-6
    p = CV.shape_outline({"points": [[5, 5]], "shape_type": "point"})
    cr = CV.crop_rect(p, 0.0, 320, 240)
    assert cr.width() >= CV.MIN_SIDE - 1 and cr.left() >= 0           # a point still gets a crop, inside the image
    out = CV.shape_outline({"points": [[400, 400], [500, 500]], "shape_type": "rectangle"})
    assert CV.crop_rect(out, 0.1, 320, 240) is None                      # wholly outside the image
    part = CV.shape_outline({"points": [[300, 200], [400, 300]], "shape_type": "rectangle"})
    assert CV.crop_rect(part, 0.0, 320, 240) == QRectF(296, 200, 24, 40)  # clipped, then MIN_SIDE inwards


def test_view_crop_from_annotations_panel(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 4))
    qtbot.waitUntil(lambda: w.current_key is not None and w.annotations_list.count() > 0)
    w.view_crop(0)
    v = w.crop_viewer
    qtbot.waitUntil(lambda: not v.isHidden())
    s0 = w.current_ann.shapes[0]
    assert v.index == 0 and v._crop is not None and not v._crop.isNull()
    b = CV.shape_outline(s0).boundingRect()
    assert v._crop.width() <= w.canvas._original.width() and v._crop.width() >= int(b.width())
    assert "#0" in v.title.text() and "car" in v.title.text()
    # padding changes the crop, outline / dim draw over it
    v.pad.setCurrentIndex(0)
    w0 = v._crop.width()
    v.pad.setCurrentIndex(v.pad.count() - 1)
    assert v._crop.width() > w0
    v.dim.setChecked(True)
    assert not v.dim_item.path().isEmpty()
    # step through the image's annotations; it wraps
    n = len(w.current_ann.shapes)
    v.step(1)
    assert v.index == 1 % n
    v.step(-1)
    v.step(-1)
    assert v.index == (n - 1)
    # follows the canvas selection and the next image
    v.follow.setChecked(True)
    w.canvas.select_shape(1)
    qtbot.wait(20)
    assert v.index == 1
    first = w.current_key
    w.a.next.trigger()
    qtbot.waitUntil(lambda: w.current_key != first)
    assert v.key == w.current_key and v._crop is not None
    # copy, zoom, close releases pixels
    v.copy_crop()
    assert not QGuiApplication.clipboard().pixmap().isNull()
    v.view.set_zoom(4.0)
    assert abs(v.view.zoom() - 4.0) < 1e-6 and not v.view.fitted
    v.view.fit()
    assert v.view.fitted
    v.close()
    assert v.pix_item.pixmap().isNull() and v._crop is None
    w.view_crop(0)                                   # the same window comes back
    assert w.crop_viewer is v and not v.isHidden()
    v.close()


def test_view_crop_without_geometry_says_so(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 4))
    qtbot.waitUntil(lambda: w.current_key is not None)
    w.current_ann.shapes.append({"label": "ghost", "points": [], "shape_type": "polygon"})
    w.view_crop(len(w.current_ann.shapes) - 1)
    v = w.crop_viewer
    assert v.message.isVisibleTo(v) and "no usable points" in v.message.text() and v._crop is None
    w.view_crop(99)                                  # out of range: a status message, nothing opens anew
    v.close()


def test_menu_has_view_crop(shell, qtbot, tmp_path):  # noqa: F811
    from tests.conftest import MESSAGES
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 4))
    qtbot.waitUntil(lambda: w.annotations_list.count() > 0)
    it = w.annotations_list.item(0)
    w._annotations_menu(w.annotations_list.visualItemRect(it).center())
    assert ("menu", "") in MESSAGES or any(m[0] == "menu" for m in MESSAGES)
    assert w.a.view_crop.shortcut().toString() == "Shift+C"
