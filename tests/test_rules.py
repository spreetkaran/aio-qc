# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Relationship rules (merged Parent / part, Never together, exact geometry, conversion, mining) and guidelines."""
from __future__ import annotations

import json
import math

import pytest

from AIO_QC.core import rules as RL

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def rect(label, x1, y1, x2, y2):
    return {"label": label, "shape_type": "rectangle", "points": [[x1, y1], [x2, y2]]}


def quad(label, cx, cy, w, h, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    pts = [[cx + dx * c - dy * s, cy + dx * s + dy * c] for dx, dy in
           ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    return {"label": label, "shape_type": "polygon", "points": pts}


PART = RL.new_rule("part", child="number", parent="plate")          # defaults: inside ≥ 90 %, not larger


def texts(vs):
    return [v.text for v in vs]


def test_number_larger_than_plate_is_found():
    plate = rect("plate", 100, 100, 300, 160)
    # a number box just a little bigger than its plate: 90 % of it is still inside, the old rule passed it
    big = rect("number", 95, 98, 306, 162)
    vs = RL.check_image([PART], [plate, big], [], 1000, 800)
    assert len(vs) == 1 and "larger than plate #0" in vs[0].text and vs[0].shapes == [1, 0]
    # a good number: nothing
    assert RL.check_image([PART], [plate, rect("number", 120, 110, 280, 150)], [], 1000, 800) == []
    # a number box enclosing the whole plate: larger AND mostly outside
    huge = rect("number", 50, 60, 400, 220)
    t = texts(RL.check_image([PART], [plate, huge], [], 1000, 800))[0]
    assert "larger than plate #0" in t and "% inside plate #0" in t
    # half off the plate / off every plate / an image without plates
    assert "only 50 % inside" in texts(RL.check_image([PART], [plate, rect("number", 250, 110, 350, 150)], [], 1, 1))[0]
    assert "not on any plate" in texts(RL.check_image([PART], [plate, rect("number", 500, 500, 560, 520)], [], 1, 1))[0]
    assert "has none" in texts(RL.check_image([PART], [rect("number", 5, 5, 9, 9)], [], 1, 1))[0]


def test_tilted_plate_uses_real_outline_and_sides():
    plate = quad("plate", 400, 300, 220, 60, 25)
    inside = quad("number", 400, 300, 180, 40, 25)                   # same tilt, smaller: fine
    assert RL.check_image([PART], [plate, inside], [], 1000, 800) == []
    # its axis-aligned box sticks out of the plate's box a lot less than the truth: exact geometry decides
    longer = quad("number", 400, 300, 240, 40, 25)                   # 109 % of the plate's long side
    t = texts(RL.check_image([PART], [plate, longer], [], 1000, 800))
    assert t and "larger than plate" in t[0] and "109 %" in t[0]
    # a text that leaves the tilted plate through its corner: box test says inside, the outline says no
    corner = quad("number", 490, 345, 60, 30, 0)
    t = texts(RL.check_image([PART], [plate, corner], [], 1000, 800))
    assert t and "inside plate #0" in t[0]


def test_parts_per_parent_and_apart_and_normalized():
    r = RL.new_rule("part", child="number", parent="plate", min=1, max=2, apart_iou=0.1)
    p1, p2 = rect("plate", 0, 0, 100, 40), rect("plate", 200, 0, 300, 40)
    n = [rect("number", 5, 5, 45, 35), rect("number", 50, 5, 95, 35), rect("number", 10, 5, 50, 35)]
    vs = RL.check_image([r], [p1, p2] + n, [], 1000, 800)
    t = texts(vs)
    assert any("plate #0 has 3 number (expected 1–2)" in x for x in t)
    assert any("plate #1 has 0 number" in x for x in t)
    assert any("overlap" in x for x in t)
    # YOLO (0..1): geometry in pixels
    W, H = 1000, 500
    norm = lambda s: dict(s, points=[[x / W, y / H] for x, y in s["points"]])
    assert texts(RL.check_image([PART], [norm(rect("plate", 100, 100, 300, 160)),
                                         norm(rect("number", 95, 98, 306, 162))], [], W, H, normalized=True))


def test_never_and_upgrade_of_old_rules(tmp_path):
    old = [
        {"id": "a1", "type": "inside", "child": "number", "parent": "plate", "min_frac": 0.8, "severity": "error"},
        {"id": "a2", "type": "contains", "parent": "plate", "child": "number", "min": 1, "max": 2, "min_frac": 0.9,
         "severity": "error"},
        {"id": "a3", "type": "siblings", "parent": "plate", "child": "number", "max_iou": 0.2, "severity": "error"},
        {"id": "a4", "type": "excludes", "a": "car", "b": "bus"},
        {"id": "a5", "type": "exclusive", "labels": ["day", "night", "dusk"]},
        {"id": "a6", "type": "count", "label": "truck", "min": 0, "max": 1},
        {"id": "a7", "type": "count", "label": "truck", "min": 0, "max": 1},          # exact duplicate
        {"id": "a8", "type": "bogus"},
    ]
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"version": 1, "rules": old}))
    rules = RL.load_rules(path)
    types = [r["type"] for r in rules]
    assert types.count("part") == 1 and types.count("never") == 2 and types.count("count") == 1
    p = next(r for r in rules if r["type"] == "part")
    assert p["need_parent"] and p["max_rel"] == 1.0 and p["min_frac"] == 0.8 and (p["min"], p["max"]) == (1, 2)
    assert p["apart_iou"] == 0.2 and p["id"] == "a1"
    assert json.loads(path.read_text())["version"] == 2                     # converted once, on disk
    never = [r for r in rules if r["type"] == "never"]
    assert RL.check_image(never, [rect("car", 0, 0, 1, 1), rect("bus", 2, 2, 3, 3)], [], 9, 9)
    assert RL.check_image(never, [], ["day", "dusk"], None, None)
    assert RL.check_image(never, [], ["day"], None, None) == []
    assert RL.load_rules(path) == rules                                     # stable


def test_mining_suggests_one_part_rule_per_pair():
    st = RL.RuleStats()
    for i in range(60):
        x = 10 * (i % 7)
        shapes = [rect("plate", x, 0, x + 100, 40), rect("number", x + 10, 5, x + 90, 35),
                  rect("car", x - 50, -50, x + 200, 120)]
        if i % 3 == 0:
            shapes.append(rect("car", x + 300, 0, x + 500, 120))
            shapes.append(rect("car", x + 305, 2, x + 505, 122))            # overlapping cars: not a rule's job
        st.add(shapes, [])
    sug = RL.suggest(st, min_support=20)
    parts = [s.rule for s in sug if s.rule["type"] == "part"]
    pairs = {(r["child"], r["parent"]) for r in parts}
    assert ("number", "plate") in pairs
    np_rule = next(r for r in parts if (r["child"], r["parent"]) == ("number", "plate"))
    assert (np_rule["min"], np_rule["max"]) == (1, 1) and np_rule["max_rel"] == 1.0
    assert not any(s.rule["type"] == "overlap" for s in sug)                 # Duplicate Shapes covers that
    assert not any(s.rule["type"] == "requires" and (s.rule["a"], s.rule["b"]) == ("number", "plate") for s in sug)
    assert len({RL.signature(s.rule) for s in sug}) == len(sug)               # nothing twice


def test_trigger_labels():
    assert RL.trigger_labels(PART) == {"number"}
    assert RL.trigger_labels(RL.new_rule("part", child="n", parent="p", min=1)) == {"n", "p"}
    assert RL.trigger_labels(RL.new_rule("count", label="truck", min=0, max=1)) == {"truck"}
    assert RL.trigger_labels(RL.new_rule("count", label="truck", min=1, max=1)) is None     # absence is a problem
    assert RL.trigger_labels(RL.new_rule("allowed", labels=["a"])) is None


def test_dialog_scan_filter_and_rule_guidelines(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import state_dir
    root = FX.labelme(tmp_path / "ds", n=12)
    # frame 5: its plate drawn bigger than the car around it
    f = root / "cam_frame_0005.json"
    d = json.loads(f.read_text())
    d["shapes"][1]["points"] = [[0, 0], [319, 0], [319, 239], [0, 239]]
    f.write_text(json.dumps(d))
    w = _open(shell, qtbot, root)
    state_dir(root).mkdir(parents=True, exist_ok=True)
    car_plate = RL.new_rule("part", child="plate", parent="car", name="Plate on its car")
    RL.save_rules(state_dir(root) / "rules.json", [car_plate, RL.new_rule("count", label="wheel", min=0, max=1)])
    w.open_tool("rules")
    dlg = w.open_dialogs["rules"]
    assert [r["type"] for r in dlg.rules] == ["part", "count"]
    # only files with a rule's class are opened
    assert len(dlg.samples_to_scan()) == len(w.session.samples)                       # every file has a plate
    dlg.rules.append(RL.new_rule("requires", a="Car ", b="truck"))                     # only frame 3 has "Car "
    saved = [r for r in dlg.rules]
    for r in dlg.rules[:2]:
        r["enabled"] = False
    assert [x.key.endswith("cam_frame_0003.json") for x in dlg.samples_to_scan()] == [True]
    dlg.rules = saved[:2]
    for r in dlg.rules:
        r["enabled"] = True
    dlg.start_scan()
    qtbot.waitUntil(lambda: dlg.worker is None or not dlg.worker.isRunning(), timeout=20000)
    qtbot.waitUntil(lambda: dlg.filter_box.count() > 1, timeout=5000)
    rows = [r for r in dlg.rows if r]
    hit = [r for r in rows if "cam_frame_0005" in r.key]
    assert hit and hit[0].cells[0] == "Plate on its car" and "larger than car" in hit[0].cells[1]
    names = [dlg.filter_box.itemData(i) for i in range(dlg.filter_box.count())]
    assert "Plate on its car" in names                                                 # filter by rule
    # duplicates are refused
    assert not dlg._accept_rule(dict(car_plate, id="zz"))
    # per-rule guideline: listed in Guidelines, shown above that rule's results
    w.session.guides.set_text(f"rule:{car_plate['id']}", "The plate box never extends past the car.")
    dlg.filter_box.setCurrentIndex(names.index("Plate on its car"))
    assert dlg.guide_card.title.text() == "Rule · Plate on its car" and dlg.guide_card.isVisibleTo(dlg)
    w.open_guidelines(f"rule:{car_plate['id']}")
    g = w.open_dialogs["guidelines"]
    scopes = [g.scopes.item(i).data(0x0100) for i in range(g.scopes.count())]
    assert f"rule:{car_plate['id']}" in scopes and g.scope() == f"rule:{car_plate['id']}"
    assert "tool:report" not in scopes and "tool:screenshots" not in scopes and "tool:duplicates" in scopes
    assert "Plate on its car" in g.header.text()
    out = tmp_path / "g.html"
    w.session.guides.export_html(out)
    assert "Rule · Plate on its car" in out.read_text()
    g.close()
    dlg.close()


def test_shape_example_draws_the_real_outline(tmp_path):
    import cv2
    import numpy as np
    from AIO_QC.core.guidelines import GuideStore
    from AIO_QC.widgets.guidelines import shape_example
    img = tmp_path / "x.png"
    cv2.imwrite(str(img), np.full((200, 300, 3), 128, np.uint8))
    st = GuideStore(tmp_path)
    circle = {"label": "wheel", "shape_type": "circle", "points": [[150, 100], [190, 100]]}
    eid = shape_example(st, "wheel", "good", circle, img, "round")
    ex = st.get("wheel")["examples"][0]
    assert eid and ex["box"] == [110.0, 60.0, 190.0, 140.0]                 # the circle's extent, not 2 points
    crop = cv2.imread(str(st.image(ex)))
    assert crop is not None and crop.shape[0] > 50


@pytest.mark.parametrize("n", [200])
def test_check_is_fast(n):
    import time
    shapes = []
    for i in range(n):
        x = (i % 20) * 50
        y = (i // 20) * 50
        shapes += [quad("plate", x + 20, y + 20, 40, 14, 10), quad("number", x + 20, y + 20, 30, 9, 10)]
    rules = [RL.new_rule("part", child="number", parent="plate", min=1, max=1, apart_iou=0.1)]
    t = time.perf_counter()
    vs = RL.check_image(rules, shapes, [], 2000, 2000)
    assert vs == [] and time.perf_counter() - t < 0.5                      # 400 shapes, exact geometry


# ---------------------------------------------------------------------------------------------------- 2: fixes, more rules
def _inside(pts, poly, tol=1e-3):
    import cv2
    import numpy as np
    pp = np.asarray(poly, np.float32)
    return all(cv2.pointPolygonTest(pp, (float(x), float(y)), True) >= -tol for x, y in pts)


def test_fixes_proposed():
    plate = rect("plate", 100, 100, 300, 160)
    v = RL.check_image([PART], [plate, rect("number", 95, 98, 306, 162)], [], 1000, 800)[0]
    assert v.fix["op"] == "set" and v.fix["points"] == [[100.0, 100.0], [300.0, 160.0]] and v.fix["shape"] == 1
    # a tilted polygon part: every vertex ends up on / inside the tilted plate, 4 points kept
    tp = quad("plate", 400, 300, 220, 60, 25)
    v = RL.check_image([PART], [tp, quad("number", 400, 300, 240, 40, 25)], [], 1000, 800)[0]
    assert len(v.fix["points"]) == 4 and _inside(v.fix["points"], tp["points"])
    after = dict(quad("number", 0, 0, 1, 1, 0), points=v.fix["points"])
    assert RL.check_image([PART], [tp, after], [], 1000, 800) == []                 # the fix fixes it
    # a stray part: delete it; same-class duplicates: delete the smaller
    assert RL.check_image([PART], [plate, rect("number", 500, 500, 560, 520)], [], 1, 1)[0].fix["op"] == "delete"
    dup = RL.new_rule("overlap", a="car", b="car", max_iou=0.5)
    v = RL.check_image([dup], [rect("car", 0, 0, 100, 100), rect("car", 1, 1, 99, 98)], [], 1, 1)[0]
    assert v.fix == {"title": "Delete the smaller one (#1)", "op": "delete", "shape": 1}
    # rectangle <-> polygon
    st = RL.new_rule("shape_type", label="plate", types=["rectangle"])
    v = RL.check_image([st], [quad("plate", 50, 50, 40, 20, 0)], [], 1, 1)[0]
    assert v.fix["type"] == "rectangle" and v.fix["shape"] == 0 and len(v.fix["points"]) == 2
    # YOLO (0..1): fixes in 0..1 too
    W, H = 1000, 500
    norm = lambda s: dict(s, points=[[x / W, y / H] for x, y in s["points"]])
    v = RL.check_image([PART], [norm(plate), norm(rect("number", 95, 98, 306, 162))], [], W, H, normalized=True)[0]
    assert [c for xy in v.fix["points"] for c in xy] == pytest.approx([0.1, 0.2, 0.3, 0.32])


def test_more_rules():
    plate = rect("plate", 100, 100, 300, 160)
    small = RL.new_rule("part", child="number", parent="plate", min_rel=0.3)
    assert "too small" in texts(RL.check_image([small], [plate, rect("number", 120, 110, 140, 120)], [], 1, 1))[0]
    aligned = RL.new_rule("part", child="number", parent="plate", max_angle=10, max_rel=0)
    tp = quad("plate", 400, 300, 220, 60, 25)
    assert RL.check_image([aligned], [tp, quad("number", 400, 300, 120, 30, 30)], [], 1, 1) == []
    assert "turned 20°" in texts(RL.check_image([aligned], [tp, quad("number", 400, 300, 120, 30, 5)], [], 1, 1))[0]
    # points as parts (keypoint on a plate)
    pt = {"label": "number", "shape_type": "point", "points": [[200, 130]]}
    assert RL.check_image([PART], [plate, pt], [], 1, 1) == []
    out = {"label": "number", "shape_type": "point", "points": [[20, 13]]}
    assert "not on any plate" in texts(RL.check_image([PART], [plate, out], [], 1, 1))[0]
    # must touch / not more than / tilt
    touch = RL.new_rule("touch", a="rider", b="bike")
    bike = rect("bike", 0, 50, 100, 100)
    assert RL.check_image([touch], [bike, rect("rider", 20, 0, 80, 60)], [], 1, 1) == []
    assert "touches no bike" in texts(RL.check_image([touch], [bike, rect("rider", 200, 0, 260, 40)], [], 1, 1))[0]
    ratio = RL.new_rule("ratio", a="plate", b="car")
    assert "2 plate but 1 car" in texts(RL.check_image([ratio], [rect("car", 0, 0, 9, 9), plate, plate], [], 1, 1))[0]
    assert RL.check_image([ratio], [rect("car", 0, 0, 9, 9), plate], [], 1, 1) == []
    tilt = RL.new_rule("tilt", label="plate", max_deg=20)
    assert "turned 35°" in texts(RL.check_image([tilt], [quad("plate", 50, 50, 80, 20, 35)], [], 100, 100))[0]
    assert RL.check_image([tilt], [quad("plate", 50, 50, 80, 20, -10)], [], 100, 100) == []
    for r in (small, touch, ratio, tilt):
        assert RL.describe(r) and RL.trigger_labels(r)


def test_mining_touch_and_ratio():
    st = RL.RuleStats()
    for i in range(60):
        shapes = [rect("car", 0, 0, 200, 100), rect("car", 300, 0, 500, 100), rect("plate", 50, 60, 90, 80),
                  rect("rider", 600, 0, 660, 80), rect("bike", 590, 50, 700, 140)]
        if i % 2:
            shapes.append(rect("plate", 350, 60, 390, 80))
        st.add(shapes, [])
    sug = RL.suggest(st, min_support=20)
    kinds = {(s.rule["type"], s.rule.get("a"), s.rule.get("b")) for s in sug}
    assert ("touch", "rider", "bike") in kinds
    assert ("ratio", "plate", "car") in kinds
    assert ("requires", "plate", "car") not in kinds and ("requires", "rider", "bike") not in kinds


def test_dialog_apply_fix_undo_and_cards(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    from AIO_QC.core.state import REVIEW_FIXED, state_dir
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    root = FX.labelme(tmp_path / "ds", n=6)
    f = root / "cam_frame_0002.json"
    d = json.loads(f.read_text())
    car = d["shapes"][0]["points"]
    x1, y1, x2, y2 = car[0][0], car[0][1], car[1][0], car[1][1]
    d["shapes"][1] = {"label": "plate", "shape_type": "rectangle", "points": [[x1 - 8, y1 + 10], [x2 + 6, y2 - 10]],
                      "flags": {}, "group_id": None, "description": ""}
    f.write_text(json.dumps(d))
    w = _open(shell, qtbot, root)
    state_dir(root).mkdir(parents=True, exist_ok=True)
    RL.save_rules(state_dir(root) / "rules.json", [RL.new_rule("part", child="plate", parent="car")])
    w.open_tool("rules")
    dlg = w.open_dialogs["rules"]
    dlg.start_scan()
    qtbot.waitUntil(lambda: dlg.worker is None or not dlg.worker.isRunning(), timeout=20000)
    qtbot.waitUntil(lambda: any(r for r in dlg.rows), timeout=5000)
    row = next(r for r in dlg.rows if r and r.key.endswith("cam_frame_0002.json"))
    assert row.cells[2].startswith("Fit inside car") and "proposed fix" in dlg.summarize()
    # a card shows the part (red), the car (dashed) and the fix (green)
    card = next(c for c in dlg.cards.model_.cards if c.row is row)
    cols = [s.color for s in card.panels[0].shapes]
    assert cols[0] == "#ffd400" and cols[-1] == "#00e676" and len(cols) == 3
    assert card.lines[0].startswith("plate → car · ") and "% size" in card.lines[0]
    assert any(line.startswith("✔ Fix: Fit inside car") for line in card.lines)
    # opening frames the part and its car, fix overlaid
    calls = []
    monkeypatch.setattr(w, "open_sample", lambda *a, **k: calls.append((a, k)))
    dlg.on_row_activated(row)
    (_key, idx), kw = calls[-1]
    assert idx == 1 and kw["overlay"][0]["text"].startswith("fix:") and kw["focus_box"][0] < x1 - 8
    # apply -> the plate fits in the car, file marked Fixed, problem gone; undo puts it back
    monkeypatch.setattr(dlg, "target_rows", lambda: [row])
    monkeypatch.setattr(dlg, "report", lambda *a, **k: None)
    dlg.apply_fixes()
    pts = json.loads(f.read_text())["shapes"][1]["points"]
    assert pts == [[x1, y1 + 10], [x2, y2 - 10]]
    assert w.session.review.status(row.key) == REVIEW_FIXED
    assert not any(r for r in dlg.rows if r and r.key.endswith("cam_frame_0002.json"))
    assert dlg.undo_btn.isEnabled()
    dlg._undo()
    assert json.loads(f.read_text())["shapes"][1]["points"] == [[x1 - 8, y1 + 10], [x2 + 6, y2 - 10]]
    dlg.close()


def test_status_bar_flagged_count(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import REVIEW_FLAG
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", n=6))
    b = w.flagged_btn
    assert b.isVisibleTo(w) and b.text() == "0 flagged" and not b.isEnabled()
    keys = [s.key for s in w.session.samples]
    w.session.set_review(keys[:3], REVIEW_FLAG)
    qtbot.waitUntil(lambda: b.text() == "3 flagged", timeout=3000)
    b.click()                                                   # only the flagged files
    assert w.file_dock.review_box.currentData() == REVIEW_FLAG
    assert sorted(w.file_dock.visible_keys()) == sorted(keys[:3])
    b.click()                                                   # back to all files
    assert w.file_dock.review_box.currentData() is None
    w.session.set_review(keys[:1], None)
    qtbot.waitUntil(lambda: b.text() == "2 flagged", timeout=3000)


# ---------------------------------------------------------------------------------------------------- 3: universal
def test_any_class_rules():
    roi = rect("roi", 0, 0, 100, 100)
    objs = [roi, rect("cat", 10, 10, 30, 30), rect("dog", 90, 90, 140, 120), {"label": "kp", "shape_type": "point",
                                                                           "points": [[50, 50]]}]
    inside_roi = RL.new_rule("part", child="*", parent="roi", max_rel=0)
    assert RL.describe(inside_roi).startswith("object is part of roi")
    v = RL.check_image([inside_roi], objs, [], 200, 200)
    assert [x.shapes[0] for x in v] == [2]                         # the dog sticks out; the roi itself is no part
    assert RL.trigger_labels(inside_roi) is None
    # no two objects of any class overlap (each pair once, never a shape with itself)
    anyov = RL.new_rule("overlap", a="*", b="*", max_iou=0.1)
    shapes = [rect("a", 0, 0, 10, 10), rect("b", 1, 1, 10, 10), rect("c", 50, 50, 60, 60)]
    v = RL.check_image([anyov], shapes, [], 99, 99)
    assert len(v) == 1 and v[0].shapes == [0, 1] and RL.describe(anyov).startswith("no two objects overlap")
    one_side = RL.new_rule("overlap", a="*", b="b", max_iou=0.1)
    assert len(RL.check_image([one_side], shapes, [], 99, 99)) == 1
    # every object is a rectangle / every object touches some other object
    rects = RL.new_rule("shape_type", label="*", types=["rectangle"])
    assert len(RL.check_image([rects], [quad("x", 5, 5, 4, 4, 0), rect("y", 0, 0, 1, 1)], [], 9, 9)) == 1
    touch_any = RL.new_rule("touch", a="person", b="*")
    assert RL.check_image([touch_any], [rect("person", 0, 0, 10, 10), rect("bicycle", 5, 5, 20, 20)], [], 99, 99) == []
    assert RL.check_image([touch_any], [rect("person", 0, 0, 10, 10)], [], 99, 99)    # alone: touches nothing
    tilt_any = RL.new_rule("tilt", label="*", max_deg=10)
    assert len(RL.check_image([tilt_any], [quad("a", 50, 50, 40, 10, 30), quad("b", 50, 50, 40, 10, 0)], [], 99, 99)) == 1


def test_rule_form_offers_any_class(qapp):
    from AIO_QC.widgets.rules_dialog import RuleForm
    for rtype, keys in RL.WILDCARD.items():
        f = RuleForm(None, RL.new_rule(rtype), ["a", "b"])
        for k in keys:
            assert f.widgets[k].findText("*") >= 0, (rtype, k)
        f.deleteLater()
    f = RuleForm(None, RL.new_rule("requires"), ["a", "b"])
    assert f.widgets["a"].findText("*") < 0                        # "an image with any object requires …" is meaningless


# ---------------------------------------------------------------------------------------------------- 4: choices from data
def _profile():
    import random
    rnd = random.Random(1)
    st = RL.RuleStats()
    for i in range(300):
        shapes = []
        for k in range(rnd.randint(1, 3)):                           # cars, each with 4 wheels (+ a person)
            x = k * 300
            shapes.append(dict(rect("car", x, 100, x + 200, 180), flags={"occluded": False}))
            for j in range(4):
                shapes.append(rect("wheel", x + 10 + j * 45, 150, x + 40 + j * 45, 178))
        shapes.append(quad("sign", 900, 60, 80, 30, rnd.uniform(-8, 8)))
        if i % 3 == 0:
            shapes.append(rect("person", 50, 40, 80, 120))
            shapes.append(rect("bicycle", 40, 90, 100, 130))
        st.add(shapes, [], 1280, 720)
    return st


def test_rank_and_recommend_from_data():
    st = _profile()
    assert RL.rank(st, "part", "parent", {"child": "wheel"})[0][0] == "car"
    assert RL.rank(st, "part", "child", {"parent": "car"})[0][0] == "wheel"
    assert RL.rank(st, "touch", "b", {"a": "person"})[0][0] in ("bicycle", "car")
    h = RL.recommend(st, "part", {"child": "wheel", "parent": "car"})
    assert h["min"].value == 4 and h["max"].value == 4 and h["need_parent"].value is True
    assert h["max_rel"].value == 1.0 and 0.5 <= h["min_frac"].value <= 1.0
    assert "apart_iou" not in h or h["apart_iou"].value <= 0.1
    h = RL.recommend(st, "count", {"label": "car"})
    assert h["min"].value == 1 and h["max"].value == 3
    h = RL.recommend(st, "tilt", {"label": "sign"})
    assert 5 <= h["max_deg"].value <= 10
    h = RL.recommend(st, "size", {"label": "wheel"})
    assert h["min_w"].value <= 30 <= h["max_w"].value and h["min_h"].value <= 28 <= h["max_h"].value
    h = RL.recommend(st, "shape_type", {"label": "sign"})
    assert h["types"].value == ["polygon"]
    h = RL.recommend(st, "attribute", {"label": "car"})
    assert h["flag"].value == "occluded" and h["value"].value == "false"
    h = RL.recommend(st, "ratio", {"a": "wheel", "b": "car"})
    assert h["per"].value == 4.0
    assert set(RL.recommend(st, "allowed", {})["labels"].value) == {"car", "wheel", "sign", "person", "bicycle"}
    for t in RL.TYPES:                                               # never fails, whatever the rule
        RL.recommend(st, t, RL.new_rule(t))


def test_form_suggests_and_applies_choices(qapp):
    from AIO_QC.widgets.rules_dialog import RuleForm
    st = _profile()
    f = RuleForm(None, RL.new_rule("part", child="wheel", parent="person", max_rel=0.0), ["car", "wheel", "person"])
    assert "Reading the dataset" in f.data_note.text()
    f.set_profile(st)
    assert "car" in f.hints["parent"].text()                        # the likely parent of wheel
    f._use("parent", "car")
    assert f.widgets["parent"].currentText() == "car"
    assert "use 4" in f.hints["max"].text()
    f.use_all_suggestions()
    v = f.value()
    assert v["min"] == 4 and v["max"] == 4 and v["max_rel"] == 1.0 and v["need_parent"] is True
    assert f.hints["max"].text() and "use" not in f.hints["max"].text()      # in use now: no link
    a = RuleForm(None, RL.new_rule("attribute", label="car"), ["car"], st)
    assert a.widgets["flag"].findText("occluded") >= 0


# ---------------------------------------------------------------------------------------------------- 5: measures, flags
def test_measures_on_violations():
    plate = rect("car", 100, 100, 300, 160)
    r = RL.new_rule("part", child="wheel", parent="car")
    v = RL.check_image([r], [plate, rect("wheel", 95, 98, 306, 162)], [], 1000, 800)[0]
    assert v.info["size"] == pytest.approx(1.07, abs=0.01) and 0.8 < v.info["inside"] < 0.95
    assert RL.measures(v.info).startswith("107 % size · ") and "% inside" in RL.measures(v.info)
    ov = RL.new_rule("overlap", a="car", b="car", max_iou=0.5)
    v = RL.check_image([ov], [rect("car", 0, 0, 100, 100), rect("car", 10, 0, 100, 100)], [], 1, 1)[0]
    assert RL.measures(v.info) == "IoU 0.90"
    cnt = RL.new_rule("count", label="car", min=0, max=1)
    v = RL.check_image([cnt], [plate, plate], [], 1, 1)[0]
    assert RL.measures(v.info) == "2 (expected 0–1)"
    tilt = RL.new_rule("tilt", label="sign", max_deg=10)
    assert RL.measures(RL.check_image([tilt], [quad("sign", 50, 50, 40, 10, 30)], [], 99, 99)[0].info) == "tilt 30°"


def test_cards_show_measures_and_flags_carry_the_reason(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    from AIO_QC.core.state import REVIEW_FLAG, state_dir
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    root = FX.labelme(tmp_path / "ds", n=6)
    for i in (1, 3):
        f = root / f"cam_frame_{i:04d}.json"
        d = json.loads(f.read_text())
        (x1, y1), (x2, y2) = d["shapes"][0]["points"]
        d["shapes"][1] = {"label": "plate", "shape_type": "rectangle", "points": [[x1 - 8, y1 + 10], [x2 + 6, y2 - 10]]}
        f.write_text(json.dumps(d))
    w = _open(shell, qtbot, root)
    state_dir(root).mkdir(parents=True, exist_ok=True)
    RL.save_rules(state_dir(root) / "rules.json", [RL.new_rule("part", child="plate", parent="car", name="On its car")])
    w.open_tool("rules")
    dlg = w.open_dialogs["rules"]
    dlg.start_scan()
    qtbot.waitUntil(lambda: dlg.worker is None or not dlg.worker.isRunning(), timeout=20000)
    qtbot.waitUntil(lambda: len([r for r in dlg.rows if r]) == 2, timeout=5000)
    card = dlg.cards.model_.cards[0]
    tags = [s.text for s in card.panels[0].shapes]
    assert tags[0].startswith("plate #1 · ") and "% size" in tags[0] and "% inside" in tags[0]
    assert tags[1] == "car #0" and card.lines[0].startswith("plate → car · ") and card.lines[0].endswith("% size +1")
    # the crop really draws them (renders without error, tag pixels in the problem colour)
    from AIO_QC.widgets.card_gallery import PanelTask
    sample = w.session.by_key[card.row.key]
    img = PanelTask(None, 0, card.panels[0], sample.image_path or root / "cam_frame_0001.jpg", 0, 1.0)._render()
    assert not img.isNull()
    # nothing ticked: Flag files flags every file found, the note names the rule and the problem
    monkeypatch.setattr(dlg, "target_rows", lambda: [])
    dlg.flag_files()
    keys = {r.key for r in dlg.rows if r}
    assert all(w.session.review.status(k) == REVIEW_FLAG for k in keys)
    note = w.session.review.get(next(iter(keys)))["note"]
    assert note.startswith("Relationship Rules: On its car — plate #1: larger than car #0")
    dlg.close()


def test_saved_view_flag_button(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from PyQt6.QtWidgets import QMessageBox
    from AIO_QC.core.state import REVIEW_FLAG
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", n=6))
    w.open_tool("views")
    dlg = w.open_dialogs["views"]
    keys = [s.key for s in w.session.samples[:2]]
    monkeypatch.setattr(dlg, "keys_of", lambda v: keys)
    dlg.cur = {"id": "tmp", "name": "Two files"}
    dlg._flag()
    assert all(w.session.review.status(k) == REVIEW_FLAG for k in keys)
    assert w.session.review.get(keys[0])["note"] == "Saved view: Two files"
    dlg.close()
