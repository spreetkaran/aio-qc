# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""LabelMe -> YOLO conversion (core.yolo_convert): shapes, clipping, merged pieces, names, layout, re-runs."""
import base64
import os
import json
from pathlib import Path

import pytest

from AIO_QC.core import yolo_convert as YC
from AIO_QC.core.dataset import DatasetType, open_dataset


def _img(path: Path, w=100, h=50):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (90, 120, 150)).save(path)


def _lm(path: Path, shapes, w=100, h=50, image="", image_data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "5.4.1", "flags": {}, "shapes": [
        {"label": s[0], "points": s[1], "shape_type": s[2], "group_id": s[3] if len(s) > 3 else None,
         "description": "", "flags": {}, **(s[4] if len(s) > 4 else {})} for s in shapes],
        "imagePath": image or path.with_suffix(".jpg").name, "imageData": image_data,
        "imageHeight": h, "imageWidth": w}))


def _lines(p: Path):
    return [ln.split() for ln in p.read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------------------------------------------- shapes
def test_each_shape_type_for_det_and_seg():
    cid = {"a": 0, "b": 1}
    shapes = [
        {"label": "a", "points": [[10, 10], [30, 20]], "shape_type": "rectangle"},
        {"label": "b", "points": [[40, 10], [60, 10], [50, 30]], "shape_type": "polygon"},
        {"label": "a", "points": [[80, 25], [90, 25]], "shape_type": "circle"},
        {"label": "a", "points": [[1, 1], [5, 5]], "shape_type": "line"},
        {"label": "a", "points": [[1, 1]], "shape_type": "point"},
        {"label": "zzz", "points": [[1, 1], [5, 5]], "shape_type": "rectangle"},
    ]
    lines, written, dropped, excluded, _m = YC.objects(shapes, 100, 50, cid, "det")
    assert len(lines) == 3 and written == {"a": 2, "b": 1}
    assert dropped == {"lines have no area": 1, "points have no area": 1} and excluded == {"zzz": 1}
    c, x, y, w, h = map(float, lines[0].split())
    assert (c, round(x, 3), round(y, 3), round(w, 3), round(h, 3)) == (0, 0.2, 0.3, 0.2, 0.2)
    c, x, y, w, h = map(float, lines[2].split())                         # the circle's box: r = 10
    assert round(w * 100) == 20 and round(h * 50) == 20
    seg, *_ = YC.objects(shapes, 100, 50, cid, "seg")
    assert len(seg[0].split()) == 1 + 8 and len(seg[1].split()) == 1 + 6     # rectangle: 4 corners; triangle: 3
    assert len(seg[2].split()) >= 1 + 32                                    # circle: a polygon of 16+ points


def test_clipped_to_the_image_and_outside_dropped():
    cid = {"a": 0}
    shapes = [{"label": "a", "points": [[-20, -10], [50, 25]], "shape_type": "rectangle"},
              {"label": "a", "points": [[120, 60], [150, 80]], "shape_type": "rectangle"}]
    lines, _w, dropped, _e, _m = YC.objects(shapes, 100, 50, cid, "det")
    assert len(lines) == 1 and dropped == {"outside the image": 1}
    _c, x, y, w, h = map(float, lines[0].split())
    assert (round(x, 3), round(y, 3), round(w, 3), round(h, 3)) == (0.25, 0.25, 0.5, 0.5)
    seg, *_ = YC.objects(shapes[:1], 100, 50, cid, "seg")
    vals = list(map(float, seg[0].split()[1:]))
    assert min(vals) >= 0 and max(vals) <= 1 and len(vals) == 8           # clipped to a proper 4-corner box


def test_pieces_of_one_object_are_merged():
    cid = {"car": 0}
    shapes = [{"label": "car", "points": [[10, 10], [20, 10], [20, 20], [10, 20]], "shape_type": "polygon", "group_id": 3},
              {"label": "car", "points": [[30, 10], [40, 10], [40, 20], [30, 20]], "shape_type": "polygon", "group_id": 3},
              {"label": "car", "points": [[60, 10], [70, 20]], "shape_type": "rectangle", "group_id": None}]
    lines, written, _d, _e, merged = YC.objects(shapes, 100, 50, cid, "det")
    assert len(lines) == 2 and written == {"car": 2} and merged == 1
    _c, x, _y, w, _h = map(float, lines[0].split())
    assert round(w * 100) == 30 and round(x * 100) == 25                    # one box around both pieces
    seg, *_ = YC.objects(shapes, 100, 50, cid, "seg")
    assert len(seg) == 2 and len(seg[0].split()) - 1 == 2 * (4 + 4 + 2)      # both rings + the bridge back
    sep, *_ = YC.objects(shapes, 100, 50, cid, "det", merge_groups=False)
    assert len(sep) == 3


def test_mask_shape_is_traced_for_seg():
    import cv2
    import numpy as np
    m = np.zeros((20, 30), np.uint8)
    m[5:15, 5:25] = 255
    ok, png = cv2.imencode(".png", m)
    shape = {"label": "a", "points": [[40, 10], [69, 29]], "shape_type": "mask",
             "mask": base64.b64encode(png.tobytes()).decode()}
    seg, written, dropped, *_ = YC.objects([shape], 100, 50, {"a": 0}, "seg")
    assert written == {"a": 1} and not dropped
    xs = list(map(float, seg[0].split()[1::2]))
    assert round(min(xs) * 100) == 45 and round(max(xs) * 100) == 64          # the drawn area, offset by the box
    det, *_ = YC.objects([shape], 100, 50, {"a": 0}, "det")
    assert len(det) == 1


def test_bad_coordinates_are_reported_not_crashing():
    shapes = [{"label": "a", "points": [[1, "x"], [5, 5]], "shape_type": "rectangle"},
              {"label": "a", "points": [[1, 1], [1, 1], [1, 1]], "shape_type": "polygon"}]
    lines, _w, dropped, *_ = YC.objects(shapes, 100, 50, {"a": 0}, "seg")
    assert not lines and dropped == {"bad coordinates": 1, "no area": 1}


# ------------------------------------------------------------------------------------------------- whole dataset
def _dataset(root: Path):
    _img(root / "cam1" / "a.jpg")
    _lm(root / "cam1" / "a.json", [("car", [[10, 10], [30, 30]], "rectangle"),
                                   ("person", [[50, 5], [60, 5], [55, 40]], "polygon"),
                                   ("no", [[70, 10], [80, 20]], "rectangle")])
    _img(root / "cam1" / "a.png")                                            # a PNG named by another JSON
    _lm(root / "cam1" / "a_png.json", [("car", [[5, 5], [15, 15]], "rectangle")], image="a.png")
    _img(root / "b.jpg")
    _lm(root / "b.json", [("tree", [[1, 1], [9, 9]], "rectangle")])          # only a label that is left out
    # an image that only exists inside its JSON
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (100, 50), (1, 2, 3)).save(buf, "PNG")
    _lm(root / "c.json", [("person", [[20, 10], [40, 30]], "rectangle")], image="missing.png",
        image_data=base64.b64encode(buf.getvalue()).decode())
    return open_dataset(root, DatasetType.LABELME)


def test_convert_flat_det(tmp_path):
    ad = _dataset(tmp_path / "ds")
    samples = ad.discover()
    out = tmp_path / "yolo"
    st = YC.convert({"": samples}, ad, tmp_path / "ds", out,
                    YC.Options(task="det", classes=["person", "car", "no"], link="copy"))
    assert st["images"]["all"] == 4 and st["background"] == 1 and st["not_included"] == {"tree": 1}
    labels = sorted(p.relative_to(out / "labels").as_posix() for p in (out / "labels").rglob("*.txt"))
    assert labels == ["b.txt", "c.txt", "cam1/a.txt", "cam1/a_png.txt"]
    assert (out / "images" / "cam1" / "a_png.png").is_file()                # the PNG keeps its extension
    a = _lines(out / "labels" / "cam1" / "a.txt")
    assert sorted(int(x[0]) for x in a) == [0, 1, 2]                         # ids follow the chosen order
    assert (out / "images" / "c.png").is_file() and _lines(out / "labels" / "c.txt")[0][0] == "0"   # decoded
    import yaml
    y = yaml.safe_load((out / "dataset.yaml").read_text())
    assert y["names"] == {0: "person", 1: "car", 2: "no"} and y["nc"] == 3     # "no" stays a name, not False
    assert y["train"] == "images" and y["val"] == "images"
    assert (out / "classes.txt").read_text().split() == ["person", "car", "no"]
    assert "tree: 1" in (out / "conversion_report.txt").read_text()


def test_convert_splits_seg_and_background_left_out(tmp_path):
    ad = _dataset(tmp_path / "ds")
    samples = ad.discover()
    out = tmp_path / "yolo"
    by = {"train": samples[:3], "val": samples[3:]}
    st = YC.convert(by, ad, tmp_path / "ds", out, YC.Options(task="seg", classes=["car", "person"],
                                                             background="skip"))
    assert st["left_out_background"] == 1
    import yaml
    y = yaml.safe_load((out / "dataset.yaml").read_text())
    assert y["train"] == "train/images" and y["val"] == "val/images"
    for p in (out / "train" / "labels").rglob("*.txt"):
        for ln in _lines(p):
            assert len(ln) % 2 == 1 and len(ln) >= 7                         # id + at least 3 points


def test_rerun_replaces_only_what_it_wrote(tmp_path):
    ad = _dataset(tmp_path / "ds")
    out = tmp_path / "yolo"
    out.mkdir()
    (out / "my_notes.txt").write_text("keep me")
    YC.convert({"": ad.discover()}, ad, tmp_path / "ds", out, YC.Options(classes=["car"]))
    assert YC.previous_output(out)["classes"] == ["car"]
    YC.remove_previous(out)
    assert sorted(p.name for p in out.iterdir()) == ["my_notes.txt"]


def test_reopens_as_yolo_with_the_same_objects(tmp_path):
    """The output is a dataset AIO QC (and YOLO trainers) read back with the same classes and counts."""
    ad = _dataset(tmp_path / "ds")
    out = tmp_path / "yolo"
    st = YC.convert({"": ad.discover()}, ad, tmp_path / "ds", out, YC.Options(classes=["person", "car", "no"]))
    yolo = open_dataset(out, DatasetType.YOLO)
    assert list(yolo.class_names) == ["person", "car", "no"]
    from collections import Counter
    got = Counter()
    for s in yolo.discover():
        got.update(yolo.load(s).label_counts())
    assert got == st["objects"]


def test_same_name_in_two_splits_does_not_clash_when_flattened(tmp_path):
    root = tmp_path / "ds"
    for sp in ("train", "val"):
        _img(root / sp / "d.jpg")
        _lm(root / sp / "d.json", [("car", [[5, 5], [15, 15]], "rectangle")])
    ad = open_dataset(root, DatasetType.LABELME)
    out = tmp_path / "yolo"
    YC.convert({"": ad.discover()}, ad, root, out, YC.Options(classes=["car"]))
    assert sorted(p.name for p in (out / "labels").iterdir()) == ["d.txt", "d_1.txt"]
    assert sorted(p.name for p in (out / "images").iterdir()) == ["d.jpg", "d_1.jpg"]


@pytest.mark.parametrize("name", ['a: b', "x # y", "yes", "100", 'quo"te'])
def test_yaml_names_survive(tmp_path, name):
    import yaml
    txt = YC.dataset_yaml(tmp_path, [""], [name, "ok"], "det")
    assert yaml.safe_load(txt)["names"][0] == name


# ------------------------------------------------------------------------------------------------- options
def test_rename_merge_and_fold_case():
    shapes = [{"label": "Car ", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"},
              {"label": "truck", "points": [[11, 1], [19, 9]], "shape_type": "rectangle"},
              {"label": "car", "points": [[21, 1], [29, 9]], "shape_type": "rectangle"}]
    ids = YC.LabelIds(["vehicle"], {"truck": "vehicle", "car": "vehicle"}, fold=True)
    lines, written, _d, excluded, _m = YC.objects(shapes, 100, 50, ids, "det")
    assert written == {"vehicle": 3} and not excluded and {ln.split()[0] for ln in lines} == {"0"}
    no_fold = YC.LabelIds(["vehicle"], {"truck": "vehicle", "car": "vehicle"})
    _l, written, _d, excluded, _m = YC.objects(shapes, 100, 50, no_fold, "det")
    assert written == {"vehicle": 2} and excluded == {"Car ": 1}


def test_min_area_and_parts_inside_wholes():
    shapes = [{"label": "plate", "points": [[10, 10], [50, 30]], "shape_type": "rectangle"},
              {"label": "line", "points": [[12, 12], [40, 20]], "shape_type": "rectangle"},       # inside
              {"label": "line", "points": [[60, 10], [90, 20]], "shape_type": "rectangle"},       # orphan
              {"label": "plate", "points": [[95, 40], [96, 41]], "shape_type": "rectangle"}]      # 1 px²
    cid = {"plate": 0, "line": 1}
    lines, written, dropped, *_ = YC.objects(shapes, 100, 50, cid, "det", min_area=2,
                                             parts=[("line", ["plate"], 0.9)])
    assert written == {"plate": 1, "line": 1}
    assert dropped == {"smaller than the minimum area": 1, "part outside its whole": 1}
    assert YC.inside_share([[[12, 12], [40, 12], [40, 20], [12, 20]]], [[[10, 10], [50, 10], [50, 30], [10, 30]]]) > .99


def test_image_policies_reencode_and_truncated_jpeg(tmp_path):
    from PIL import Image
    root = tmp_path / "ds"
    root.mkdir()
    Image.new("RGB", (100, 50), (9, 9, 9)).save(root / "g.gif")
    _lm(root / "g.json", [("car", [[1, 1], [9, 9]], "rectangle")], image="g.gif")
    _img(root / "t.jpg")
    raw = (root / "t.jpg").read_bytes()
    (root / "t.jpg").write_bytes(raw[:-2])                       # no end marker
    _lm(root / "t.json", [("car", [[1, 1], [9, 9]], "rectangle"), ("tree", [[1, 1], [5, 5]], "rectangle")])
    _img(root / "b.jpg")
    _lm(root / "b.json", [("car", [[1, "x"], [9, 9]], "rectangle"), ("car", [[1, 1], [9, 9]], "rectangle")])
    ad = open_dataset(root, DatasetType.LABELME)
    out = tmp_path / "y"
    st = YC.convert({"": ad.discover()}, ad, root, out, YC.Options(classes=["car"], link="hardlink"))
    assert (out / "images" / "g.png").is_file()                  # GIF re-encoded
    assert os.stat(out / "images" / "t.jpg").st_ino != os.stat(root / "t.jpg").st_ino     # copied, not linked
    assert st["verify"]["problems"] == [] and st["verify"]["objects"] == st["objects"]
    out2 = tmp_path / "y2"
    st = YC.convert({"": ad.discover()}, ad, root, out2, YC.Options(classes=["car"], left_out="skip", broken="skip"))
    assert st["left_out_images"] == {"has a label that is not written": 1, "has a broken shape": 1}
    assert sorted(p.name for p in (out2 / "labels").iterdir()) == ["g.txt"]
