# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""2.3.0: Labels dock rows never stay highlighted; Fix in Editor can show only selected labels."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt

from AIO_QC.core.edit_session import EditSession
from AIO_QC.core.editors import LABELME

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_labels_dock_deselects(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    lst = w.unique_labels
    qtbot.waitUntil(lambda: lst.count() >= 3, timeout=5000)
    it = lst.item(0)
    rect = lst.visualItemRect(it)
    qtbot.mouseClick(lst.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert lst.selectedItems() == [it]
    it.setCheckState(Qt.CheckState.Unchecked)                  # turning a label off …
    assert it.data(Qt.ItemDataRole.UserRole) in w.hidden_labels
    qtbot.waitUntil(lambda: not lst.selectedItems() and lst.currentItem() is None, timeout=2000)   # … deselects
    it.setCheckState(Qt.CheckState.Checked)
    # empty space and Esc clear it too
    qtbot.mouseClick(lst.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert lst.selectedItems()
    empty = QPoint(5, lst.visualItemRect(lst.item(lst.count() - 1)).bottom() + 4)
    if lst.itemAt(empty) is None and empty.y() < lst.viewport().height():
        qtbot.mouseClick(lst.viewport(), Qt.MouseButton.LeftButton, pos=empty)
        assert not lst.selectedItems()
    qtbot.mouseClick(lst.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    qtbot.keyClick(lst, Qt.Key.Key_Escape)
    assert not lst.selectedItems() and lst.currentItem() is None
    # "all labels" off and "show only this label" leave nothing highlighted
    qtbot.mouseClick(lst.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    w.toggle_all_labels(False)
    qtbot.waitUntil(lambda: not lst.selectedItems(), timeout=2000)
    w.toggle_all_labels(True)
    w._solo_label(lst.item(0).data(Qt.ItemDataRole.UserRole))
    qtbot.waitUntil(lambda: not lst.selectedItems(), timeout=2000)
    w.toggle_all_labels(True)


def _sync(es, s, e):
    """What the dialog does for one saved file."""
    sample = s.by_key[e.key]
    upd = es.build_update(e, s.base_adapter, sample)
    if upd is None:
        return None
    es.backup_original(e, sample)
    s.locked.pop(e.key, None)
    s.write_external(e.key, lambda ann: (upd.apply(ann), True)[1])
    es.after_write(e, sample, upd.summary)
    return upd


def test_editor_only_selected_labels_labelme(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    s = w.session
    key = next(k for k in s.by_key if "0003" in k)               # car, plate, wheel, "Car "
    orig = json.loads(Path(s.by_key[key].ann_path).read_text())
    es = EditSession.create(s.root, [s.by_key[key]], s.base_adapter, LABELME, [], "none", "labelme 5",
                            label_hints=["car", "plate", "wheel"], only_labels=["plate"])
    e = es.entries[0]
    mirror = es.work / e.mirror
    data = json.loads(mirror.read_text())
    assert [x["label"] for x in data["shapes"]] == ["plate"]       # the editor sees only the selected label
    assert e.hidden == [0, 2, 3] and "3 shape(s) of other labels" in e.message
    # resumed sessions keep the filter
    assert EditSession.load(es.work, s.root).only_labels == ["plate"]
    # saved without changes -> nothing written
    es.poll(); es.poll()
    assert _sync(es, s, e) is None
    # move the plate and add a second one
    data["shapes"][0]["points"][0] = [1, 1]
    data["shapes"].append({"label": "plate", "points": [[2, 2], [8, 2], [8, 8]], "shape_type": "polygon",
                           "flags": {}, "group_id": None, "description": ""})
    mirror.write_text(json.dumps(data))
    assert _sync(es, s, e) is not None
    out = json.loads(Path(s.by_key[key].ann_path).read_text())
    assert [x["label"] for x in out["shapes"]] == ["car", "plate", "plate", "wheel", "Car "]
    assert out["shapes"][0] == orig["shapes"][0] and out["shapes"][3] == orig["shapes"][2]   # hidden: untouched
    assert out["shapes"][1]["points"][0] == [1, 1]
    # relabel a plate to a hidden label: no duplicate on the next save
    data["shapes"][1]["label"] = "car"
    mirror.write_text(json.dumps(data))
    _sync(es, s, e)
    data["shapes"][0]["points"][0] = [3, 3]
    mirror.write_text(json.dumps(data))
    _sync(es, s, e)
    out = json.loads(Path(s.by_key[key].ann_path).read_text())
    assert [x["label"] for x in out["shapes"]] == ["car", "plate", "car", "wheel", "Car "]
    assert out["shapes"][1]["points"][0] == [3, 3]
    # delete every shown shape: the hidden ones stay
    data["shapes"] = []
    mirror.write_text(json.dumps(data))
    _sync(es, s, e)
    out = json.loads(Path(s.by_key[key].ann_path).read_text())
    assert [x["label"] for x in out["shapes"]] == ["car", "wheel", "Car "]
    es.finish()


def test_editor_only_selected_labels_yolo(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.yolo(tmp_path / "ds"))
    s = w.session
    key = next(iter(s.by_key))
    es = EditSession.create(s.root, [s.by_key[key]], s.base_adapter, LABELME, ["car", "plate"], "none",
                            "labelme 5", only_labels=["car"])
    e = es.entries[0]
    mirror = es.work / e.mirror
    data = json.loads(mirror.read_text())
    assert [x["label"] for x in data["shapes"]] == ["car"]
    data["shapes"][0]["points"] = [[10, 10], [50, 40]]
    mirror.write_text(json.dumps(data))
    assert _sync(es, s, e) is not None
    lines = Path(s.by_key[key].ann_path).read_text().split("\n")
    assert lines[0].startswith("0 ") and lines[0] != "0 0.3 0.33 0.3 0.33"
    assert lines[1] == "1 0.6 0.6 0.1 0.08"                        # the hidden plate keeps its original tokens
    es.finish()


def test_editor_dialog_label_filter(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets.editor_session_dialog import EditorSessionDialog
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    w.hidden_labels.add("wheel")
    dlg = EditorSessionDialog(w, keys=[w.current_key])
    qtbot.addWidget(dlg)
    assert dlg.selected_labels() is None and not dlg.label_box.isVisibleTo(dlg)
    dlg.only_sel.setChecked(True)
    assert dlg.label_box.isVisibleTo(dlg)
    picked = dlg.selected_labels()
    assert "car" in picked and "plate" in picked and "wheel" not in picked      # preset from the Labels dock
    dlg._tick_labels(False)
    assert dlg.selected_labels() == [] and not dlg.start_btn.isEnabled()
    dlg.label_search.setText("pla")
    dlg._tick_labels(True)                                         # only the filtered rows
    assert dlg.selected_labels() == ["plate"]
    dlg._labels_from_dock()
    assert "wheel" not in dlg.selected_labels() and "car" in dlg.selected_labels()
    w.hidden_labels.discard("wheel")
    dlg.close()
