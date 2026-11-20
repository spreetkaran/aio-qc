# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Duplicate & Overlapping Shapes: exact copies / marked twice / overlap / inside, each pair once (copies grouped),
parts of another class are not mistakes, expected class pairs are skipped (persisted per dataset), either side of a
pair can be deleted. Poor box fit (Model Check): side-by-side fit against the model's box."""
import json
import os

from PyQt6.QtWidgets import QMessageBox

from AIO_QC.core.checks import find_overlaps, find_shape_pairs
from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def R(label, x1, y1, x2, y2):
    return {"label": label, "shape_type": "rectangle", "points": [[x1, y1], [x2, y2]], "flags": {},
            "group_id": None, "description": ""}


def test_find_overlaps_kinds():
    shapes = [R("car", 0, 0, 100, 100), R("car", 2, 1, 101, 100),          # marked twice
              R("car", 50, 0, 150, 100),                                   # overlaps both (IoU ~0.33)
              R("plate", 10, 10, 30, 20),                                  # a part: not a mistake
              R("car", 20, 20, 40, 40),                                    # a car inside a car
              R("person", 300, 0, 400, 100), R("bag", 320, 20, 420, 120)]  # overlap of other classes
    got = {(a, b): kind for a, b, _iou, _ins, kind in find_overlaps(shapes)}
    assert got[(0, 1)] == "double"
    assert got[(0, 2)] == "overlap"
    assert got[(0, 4)] == "inside"
    assert not any(3 in k for k in got)                                    # plate on a car
    assert got[(5, 6)] == "overlap"
    assert (5, 6) not in {(a, b) for a, b, *_ in find_overlaps(shapes, ignore={("bag", "person")})}
    assert any(3 in (a, b) for a, b, *_ in find_overlaps(shapes, nested_other_class=True))


def test_shape_pairs_group_copies_and_list_each_pair_once():
    shapes = [R("car", 0, 0, 100, 100), R("car", 2, 1, 101, 100),          # marked twice
              R("car", 50, 0, 150, 100),                                   # overlaps the group
              R("plate", 10, 10, 30, 20),                                  # a part
              R("car", 20, 20, 40, 40),                                    # inside
              R("truck", 0, 0, 100, 100),                                  # an exact copy with another label
              R("car", 1, 0, 100, 100)]                                    # a third copy
    got = find_shape_pairs(shapes)
    pairs = {(a, b): k for a, b, _i, _s, k in got}
    assert pairs[(0, 5)] == "exact" and pairs[(0, 1)] == "double" and pairs[(0, 6)] == "double"
    assert (1, 6) not in pairs and (5, 6) not in pairs and (1, 5) not in pairs      # no copy-to-copy pairs
    assert pairs[(0, 2)] == "overlap" and sum(k == "overlap" for k in pairs.values()) == 1  # once per group
    assert pairs[(0, 4)] == "inside" and not any(3 in k for k in pairs)
    same = {(a, b) for a, b, *_ in find_shape_pairs(shapes, same_class_only=True)}
    assert (0, 5) not in same and (0, 1) in same


def test_pair_card_renders():
    from AIO_QC.widgets.card_gallery import PanelTask, _Signals
    from AIO_QC.widgets.qc_basic import pair_panel
    from PyQt6.QtGui import QImage
    import tempfile
    img = QImage(400, 300, QImage.Format.Format_RGB32)
    img.fill(0xff808080)
    path = os.path.join(tempfile.mkdtemp(), "x.png")
    img.save(path)
    a = [(100, 100), (200, 100), (200, 180), (100, 180)]
    b = [(150, 120), (260, 120), (260, 210), (150, 210)]
    for kind in ("double", "overlap"):
        out = PanelTask(_Signals(), 0, pair_panel("k", a, b, "car", "car", False, kind, 0, 1), path, 0, 1.0)._render()
        assert not out.isNull() and out.width() > 0


def test_loose_and_tight_boxes_found_when_iou_is_high():
    from AIO_QC.core.model_check import POOR_FIT, fit_kind, fit_text, match
    gt = [(0, "car", (90, 100, 220, 200)),       # right side 20 % past the object, IoU ≈ 0.77: was never reported
          (1, "car", (300, 300, 400, 400)),       # fits
          (2, "car", (515, 500, 600, 600))]       # left side cuts 5 % (within tolerance)
    preds = [("car", 0.9, (100, 100, 200, 200), "m"), ("car", 0.9, (302, 301, 399, 400), "m"),
             ("car", 0.9, (510, 500, 600, 600), "m")]
    got = [f for f in match(gt, preds) if f.kind == POOR_FIT]
    assert [f.gt_index for f in got] == [0]
    assert fit_kind(got[0].sides, 0.1) == "loose" and "right 20 %" in fit_text(got[0].sides, 0.1)
    tight = match([(0, "car", (100, 100, 180, 200))], [("car", 0.9, (100, 100, 200, 200), "m")])
    assert tight[0].kind == POOR_FIT and fit_kind(tight[0].sides, 0.1) == "tight"
    assert not [f for f in match(gt, preds, fit_tol=None) if f.kind == POOR_FIT]       # off: as before
    low = [("car", 0.3, (100, 100, 200, 200), "m")]
    assert not match(gt[:1], low, conf_thr=0.25)            # a doubtful prediction is not trusted for the fit


def _dataset(root):
    FX.labelme(root)
    for i in (1, 4):
        p = os.path.join(root, f"cam_frame_{i:04d}.json")
        d = json.load(open(p))
        (x1, y1), (x2, y2) = d["shapes"][0]["points"]
        d["shapes"].append(R("car", x1 + 3, y1 + 2, x2 + 3, y2 + 2))
        d["shapes"].append(R("person", x2 + 60, 10, x2 + 120, 80))
        d["shapes"].append(R("bag", x2 + 80, 30, x2 + 140, 100))
        json.dump(d, open(p, "w"))
    return root


def test_overlaps_dialog_flow(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    w = _open(shell, qtbot, _dataset(tmp_path / "ds"))
    w.qc_actions["duplicates"].trigger()
    qtbot.waitUntil(lambda: "duplicates" in w.open_dialogs, timeout=3000)
    d = w.open_dialogs["duplicates"]
    assert "overlaps" not in w.qc_actions                 # one check since the merge
    d.start_scan()
    qtbot.waitUntil(lambda: len(d.finding_list()) == 4, timeout=10000)
    kinds = sorted(r.cells[0] for r in d.visible_rows())
    assert kinds == ["Marked twice", "Marked twice", "Overlap", "Overlap"]
    assert d.cards.model().rowCount() == 4
    # person ↔ bag is normal here: skip it from now on
    bag = [r for r in d.visible_rows() if "bag" in (r.cells[1], r.cells[2])]
    d.table.clearSelection()
    for r in range(d.table.rowCount()):
        if d._row_at(r) is bag[0]:
            d.table.selectRow(r)
    d._expect()
    assert len(d.finding_list()) == 2
    from AIO_QC.core.state import load_overlap_ignores
    assert load_overlap_ignores(w.session.root) == {("bag", "person")}
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None or not d.worker.isRunning(), timeout=10000)
    qtbot.waitUntil(lambda: len(d.finding_list()) == 2, timeout=5000)
    # delete the smaller of every "marked twice" pair
    for r in range(d.table.rowCount()):
        d.table.selectRow(r) if r == 0 else None
    d.mark_all = None
    for row in d.visible_rows():
        d.set_marked(row, True)
    d.delete_shape_rows(d.target_rows(), "shape(s)")
    qtbot.waitUntil(lambda: len(d.finding_list()) == 0, timeout=10000)
    shapes = json.load(open(os.path.join(w.session.root, "cam_frame_0001.json")))["shapes"]
    assert sum(s["label"] == "car" for s in shapes) == 1


def test_poor_fit_lives_in_model_check_only(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.app import QC_TOOLS
    from AIO_QC.core.model_check import POOR_FIT
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    assert not any(t.key == "loose_boxes" for t in QC_TOOLS) and "loose_boxes" not in w.qc_actions
    w.qc_actions["model_check"].trigger()
    qtbot.waitUntil(lambda: "model_check" in w.open_dialogs, timeout=3000)
    d = w.open_dialogs["model_check"]
    d.show_kind(POOR_FIT)                          # no run yet: says what is needed, keeps the wish
    assert d._want_kind == POOR_FIT and "model" in d.status.text()
