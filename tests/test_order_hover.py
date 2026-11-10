# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Natural file order per folder / split (file list), and no shape hover highlight while a box is drawn."""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest

from AIO_QC.core.dataset import DatasetType, open_dataset
from AIO_QC.core.io import natural_key

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_natural_key():
    names = ["f10.jpg", "f2.jpg", "F1.jpg", "f02.jpg", "a.jpg", "f1a.jpg", "f1.jpg", "10", "9", "img (10).png",
             "img (2).png", "fa1", "f", "B.jpg", "frame_000100.json", "frame_99.json"]
    assert sorted(names, key=natural_key) == [
        "9", "10", "a.jpg", "B.jpg", "f", "F1.jpg", "f1.jpg", "f1a.jpg", "f02.jpg", "f2.jpg", "f10.jpg", "fa1",
        "frame_99.json", "frame_000100.json", "img (2).png", "img (10).png"]
    assert natural_key("x" + "9" * 40) > natural_key("x" + "9" * 39)          # long numbers still by value


def _labelme_tree(root: Path):
    for split in ("test", "train", "val"):
        for cam in ("cam10", "cam2"):
            d = root / split / cam
            d.mkdir(parents=True)
            for i in (1, 2, 10, 11, 3):
                (d / f"frame_{i}.json").write_text(json.dumps({"shapes": [], "imagePath": f"frame_{i}.jpg",
                                                                "imageData": None}))
    for n in ("top_10", "top_9"):
        (root / f"{n}.json").write_text(json.dumps({"shapes": [], "imagePath": "x.jpg"}))
    (root / "misc" / "train").mkdir(parents=True)            # a lone train/ is an ordinary folder
    (root / "misc" / "train" / "a.json").write_text(json.dumps({"shapes": [], "imagePath": "a.jpg"}))
    return root


def _rel(ad, root, samples, attr="ann_path"):
    return [Path(getattr(s, attr)).relative_to(root).as_posix() for s in samples]


def test_labelme_order_natural_and_splits(tmp_path):
    root = _labelme_tree(tmp_path / "ds")
    ad = open_dataset(root, DatasetType.LABELME)
    ss = ad.discover()
    rel = _rel(ad, root, ss)
    assert rel[:2] == ["top_9.json", "top_10.json"]                            # files first, natural
    assert rel[-1] == "misc/train/a.json"            # an ordinary folder: after the splits (lone train/: no split)
    frames = [r for r in rel if r.startswith(("train/", "val/", "test/"))]
    assert [r.split("/")[0] for r in frames[::10]] == ["train", "val", "test"]  # splits: train → val → test
    assert frames[:10] == [f"train/{c}/frame_{i}.json" for c in ("cam2", "cam10") for i in (1, 2, 3, 10, 11)]
    keys = [ad.order_key(s) for s in ss]
    assert keys == sorted(keys)                                                # live inserts use the same order
    # a file that appears later lands where discovery would have put it
    new = root / "val" / "cam2" / "frame_4.json"
    new.write_text(json.dumps({"shapes": [], "imagePath": "frame_4.jpg"}))
    s, structural = ad.sample_for_path(new)
    assert not structural
    i = bisect.bisect_right(keys, ad.order_key(s))
    assert rel[i - 1] == "val/cam2/frame_3.json" and rel[i] == "val/cam2/frame_10.json"


def test_yolo_orders(tmp_path):
    root = tmp_path / "y"
    for split in ("val", "train"):
        for sub in ("images", "labels"):
            (root / sub / split).mkdir(parents=True)
        for i in (1, 10, 2):
            FX._img(root / "images" / split / f"img{i}.jpg", w=32, h=24)
            (root / "labels" / split / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text("names:\n  0: car\n")
    ad = open_dataset(root, DatasetType.YOLO)
    ss = ad.discover()
    assert _rel(ad, root, ss, "image_path") == [f"images/{s}/img{i}.jpg" for s in ("train", "val") for i in (1, 2, 10)]
    keys = [ad.order_key(s) for s in ss]
    assert keys == sorted(keys)
    root2 = FX.yolo(tmp_path / "ys", n=12)                                     # <split>/images layout
    ad2 = open_dataset(root2, DatasetType.YOLO_SPLIT)
    ss2 = ad2.discover()
    names = [Path(s.image_path).name for s in ss2]
    assert names[:12] == [f"train_{i}.jpg" for i in range(12)] and names[12] == "val_0.jpg"
    keys2 = [ad2.order_key(s) for s in ss2]
    assert keys2 == sorted(keys2)


def test_file_list_and_folder_filter_are_natural(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds", n=12)
    for cam in ("cam10", "cam2"):
        (root / cam).mkdir()
        for i in (1, 10, 2):
            src = root / f"cam_frame_{i:04d}.json"
            d = json.loads(src.read_text())
            d["imagePath"] = f"../cam_frame_{i:04d}.jpg"
            (root / cam / f"shot_{i}.json").write_text(json.dumps(d))
    w = _open(shell, qtbot, root)
    m = w.file_dock.model
    shown = [Path(m.samples[r].ann_path) for r in range(m.rowCount())]
    tail = [x for x in shown if x.name.startswith("shot_")]
    assert [[x.parent.name, x.name] for x in tail] == \
        [[c, f"shot_{i}.json"] for c in ("cam2", "cam10") for i in (1, 2, 10)]
    assert w.session.folders == ["cam2", "cam10"]


def _hover(c, pos):
    """A plain mouse move with no button held (QTest's mouseMove carries any button state an earlier test left)."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication
    vp = c.viewport()
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(vp.mapToGlobal(pos)), Qt.MouseButton.NoButton,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)


def test_no_hover_highlight_while_drawing(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None and bool(w.canvas.shape_groups), timeout=10000)
    w.resize(1100, 800)
    qtbot.wait(50)
    c = w.canvas
    c.animate_to(2.5, center=QPointF(90, 80), duration=0)
    w.a.select.trigger()
    g = c.shape_groups[0]                                                      # the "car" box
    r = g.shape_item.sceneBoundingRect()
    inside = c.mapFromScene(r.center())
    _hover(c, inside)
    qtbot.waitUntil(lambda: c.hover_group is g, timeout=2000)
    assert g.hovered
    drawn = []
    c.boxDrawn.connect(drawn.append)
    c._start_drawing()                                                         # B
    assert c.hover_group is None and not g.hovered                            # highlight gone at once
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, inside)
    assert not g.hovered
    for k in range(1, 6):                                                      # a small box inside the shape
        QTest.mouseMove(c.viewport(), inside + QPoint(3 * k, 2 * k))
        assert c.hover_group is None and not g.hovered
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       inside + QPoint(15, 10))
    assert drawn and drawn[0].width() > 0
    # a press that starts any other gesture (here: dragging the selected shape's handle area) clears it too
    for dlg in list(w.open_dialogs.values()):
        dlg.close()
    _hover(c, c.mapFromScene(c.shape_groups[0].shape_item.sceneBoundingRect().center()))
    qtbot.waitUntil(lambda: c.hover_group is not None, timeout=2000)
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     c.mapFromScene(c.shape_groups[0].shape_item.sceneBoundingRect().center()))
    assert c.hover_group is None
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       c.mapFromScene(c.shape_groups[0].shape_item.sceneBoundingRect().center()))
    _ = np  # noqa
