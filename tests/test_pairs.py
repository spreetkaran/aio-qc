# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Is a file really LabelMe / YOLO (and what it looks like when not), and Image–Label Pairs: every image with its
label file — LabelMe next to / pointed at, YOLO in mirrored images/ and labels/ folders — with partners found and
safe fixes (applied, undone)."""
import json
import os
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from AIO_QC.core.formats import labelme_report, yolo_report
from AIO_QC.core.pairs import labelme_pairs, yolo_pairs
from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _img(p):
    from PIL import Image
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.new("RGB", (32, 24)).save(p)


def _lm(p, ip, **extra):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = {"version": "5.4.1", "flags": {}, "shapes": [], "imagePath": ip, "imageData": None, "imageHeight": 24,
         "imageWidth": 32}
    d.update(extra)
    json.dump(d, open(p, "w"))


def test_labelme_or_not():
    good = {"version": "5.4.1", "flags": {}, "imagePath": "a.jpg", "imageData": None, "imageHeight": 24,
            "imageWidth": 32, "shapes": [{"label": "car", "points": [[1, 2], [3, 4]], "shape_type": "rectangle",
                                          "group_id": None, "flags": {}}]}
    r = labelme_report(good)
    assert r.ok and not r.problems and not r.items
    bad = dict(good, imageWidth=0, shapes=[{"label": 5, "points": [[1, 2]], "shape_type": "rectangle"}, "x",
                                           {"label": "a", "points": [[1, "q"], [2, 3], [4, 5]], "shape_type": "polygon"}])
    r = labelme_report(bad)
    assert r.ok and any("imageWidth" in t for _s, t in r.problems)
    texts = [t for _i, _s, t in r.items]
    assert any("not text" in t for t in texts) and any("needs 2" in t for t in texts)
    assert any("not an object" in t for t in texts) and any("[x, y]" in t for t in texts)
    for data, what in (({"images": [], "annotations": [], "categories": []}, "COCO"),
                       ([{"data": {}, "annotations": []}], "Label Studio"), ({"objects": [], "size": {}}, "Supervisely"),
                       ({"_via_img_metadata": {}}, "VIA"), ([], "empty JSON list")):
        r = labelme_report(data)
        assert not r.ok and what in r.looks_like


def test_yolo_or_not():
    assert yolo_report("0 0.5 0.5 0.2 0.3\n").ok and yolo_report("").ok           # empty = background
    for text, what in (("car 10 20 30 40\n", "class names"), ("0 100 200 50 60\n", "pixel coordinates"),
                       ("car\nperson\n", "class names"), ("{}", "JSON"), ("<annotation/>", "XML"),
                       ("10 20 30 20 30 40 10 40 plane 0\n", "DOTA"),
                       ("dog 0.00 0 -10 1 2 3 4 1.5 1.6 1.7 1 2 3 0.1\n", "KITTI")):
        r = yolo_report(text)
        assert not r.ok and what in r.looks_like, (text, r.looks_like)
    assert not yolo_report("car\n", name="classes.txt").ok
    r = yolo_report("0 0.5 0.5 0 0.3\n0 0.1 0.2 0.3\n0.5 0.1 0.1 0.1 0.1\n3 0.5 0.5 0.1 0.1\n", n_classes=2)
    t = [x for _i, _s, x in r.items]
    assert len(t) == 4 and "zero" in t[0] and "value(s)" in t[1] and "whole number" in t[2] and "class list" in t[3]
    assert any("mixed" in x for _s, x in yolo_report("0 .5 .5 .1 .1\n0 .1 .1 .2 .1 .3 .4\n").problems)


def test_labelme_pairs(tmp_path):
    r = tmp_path / "lm"
    _img(r / "a.jpg"); _lm(r / "a.json", "a.jpg")                               # paired
    _lm(r / "b.json", "b.jpg")                                                  # no image anywhere
    _img(r / "sub" / "c.jpg"); _lm(r / "c.json", "c.jpg")                       # image elsewhere
    _lm(r / "d.json", "d.jpg", imageData="aGk=")                                # only embedded
    _img(r / "e.jpg")                                                           # no JSON
    _img(r / "f.jpg"); _lm(r / "f.json", "f.jpg"); _lm(r / "f2.json", "f.jpg")  # one image, two JSONs
    _img(r / "h.jpg"); _lm(r / "h.json", "zzz.jpg")                             # imagePath wrong, same name next to it
    json.dump({"images": [], "annotations": []}, open(r / "coco.json", "w"))     # not LabelMe
    rep = labelme_pairs(r)
    got = {(Path(f.path).name, f.kind): f for f in rep.findings}
    assert got[("b.json", "json_no_image")].fix == "trash"
    assert got[("c.json", "json_no_image")].fix_arg == "sub/c.jpg"
    assert ("d.json", "image_only_in_json") in got and ("e.jpg", "image_no_json") in got
    assert ("f.json", "shared_image") in got and ("f2.json", "shared_image") in got
    assert ("f2.json", "path_other_name") not in got                           # said once, not twice
    assert got[("h.json", "path_other_name")].fix_arg == "h.jpg"
    assert "COCO" in got[("coco.json", "not_labelme")].detail
    assert ("c.jpg", "image_no_json") not in got                               # proposed from c.json's side
    assert rep.matched == 4                                                    # a, f, f2, h


def test_yolo_pairs(tmp_path):
    r = tmp_path / "yo"
    for sp in ("train", "val"):
        (r / sp / "labels").mkdir(parents=True)
    L = "0 0.5 0.5 0.2 0.2\n"
    _img(r / "train/images/a.jpg"); (r / "train/labels/a.txt").write_text(L)                     # paired
    _img(r / "train/images/b.jpg")                                                              # background?
    _img(r / "train/images/c.jpg"); (r / "val/labels/c.txt").write_text(L)                      # label in other split
    (r / "train/labels/d.txt").write_text(L)                                                    # no image anywhere
    _img(r / "val/images/E.jpg"); (r / "val/labels/e.txt").write_text(L)                        # case
    _img(r / "val/images/f.jpg"); (r / "val/images/f.txt").write_text(L)                        # label in images/
    _img(r / "val/images/g.jpg"); _img(r / "val/images/g.png"); (r / "val/labels/g.txt").write_text("car 1 2 3 4\n")
    (r / "train/labels/classes.txt").write_text("car\n")                                         # not a label file
    rep = yolo_pairs([("train", r / "train"), ("val", r / "val")], r, 1)
    got = {(Path(f.path).name, f.kind): f for f in rep.findings}
    assert got[("b.jpg", "image_no_label")].fix == "empty_label"
    c = got[("c.jpg", "image_no_label")]
    assert c.fix == "move_label" and c.fix_arg[0].endswith(os.path.join("val", "labels", "c.txt"))
    assert ("c.txt", "label_no_image") not in got                              # said once
    assert got[("d.txt", "label_no_image")].fix == "trash"
    assert got[("e.txt", "case_mismatch")].fix_arg.endswith("E.txt")
    assert got[("f.txt", "label_in_images")].fix == "move_label"
    assert ("g.txt", "several_images") in got and "class names" in got[("g.txt", "not_yolo")].detail
    assert not any(Path(f.path).name == "classes.txt" for f in rep.findings)
    assert rep.matched == 3                                                    # a.jpg, g.jpg, g.png


def test_pairs_dialog_fixes_and_undo(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    root = FX.yolo(tmp_path / "ds")
    os.remove(os.path.join(root, "train", "labels", "train_1.txt"))                              # background?
    os.replace(os.path.join(root, "train", "labels", "train_2.txt"),
               os.path.join(root, "val", "labels", "train_2.txt"))                               # wrong split
    w = _open(shell, qtbot, root)
    w.qc_actions["pairs"].trigger()
    qtbot.waitUntil(lambda: "pairs" in w.open_dialogs, timeout=3000)
    d = w.open_dialogs["pairs"]
    qtbot.waitUntil(lambda: d.worker is None or not d.worker.isRunning(), timeout=10000)
    qtbot.waitUntil(lambda: len(d.visible_rows()) == 2, timeout=5000)
    kinds = sorted(r.payload.kind for r in d.visible_rows())
    assert kinds == ["image_no_label", "image_no_label"]
    assert "paired" in d.summarize() and "14 of 16" in d.summarize()
    monkeypatch.setattr(d, "report", lambda *a, **k: None)
    for r in d.visible_rows():
        d.set_marked(r, True)
    d.apply_fixes()
    t1 = os.path.join(root, "train", "labels", "train_1.txt")
    t2 = os.path.join(root, "train", "labels", "train_2.txt")
    assert os.path.isfile(t1) and open(t1).read() == ""                         # empty label: background
    assert os.path.isfile(t2) and not os.path.exists(os.path.join(root, "val", "labels", "train_2.txt"))
    qtbot.waitUntil(lambda: d.worker is None or not d.worker.isRunning(), timeout=10000)
    qtbot.waitUntil(lambda: len(d.visible_rows()) == 0, timeout=5000)
    d.undo_fixes()
    assert not os.path.exists(t1) and os.path.isfile(os.path.join(root, "val", "labels", "train_2.txt"))


def test_validation_says_what_a_wrong_file_is(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.checks import ValidationOptions, validate_sample
    root = FX.labelme(tmp_path / "lm", n=3)
    json.dump({"images": [], "annotations": [], "categories": []}, open(os.path.join(root, "coco.json"), "w"))
    w = _open(shell, qtbot, root)
    s = w.session.by_key[str(Path(root) / "coco.json")]
    iss = validate_sample(w.session.adapter, s, ValidationOptions())
    assert iss[0].category == "Not LabelMe format" and "COCO" in iss[0].detail
    root2 = FX.yolo(tmp_path / "yo", n=2)
    Path(root2, "train", "labels", "train_0.txt").write_text("car 10 20 30 40\n")
    w2 = _open(shell, qtbot, root2)
    s2 = next(x for x in w2.session.samples if x.name == "train_0.jpg")
    iss = validate_sample(w2.session.adapter, s2, ValidationOptions())
    assert len(iss) == 1 and iss[0].category == "Not YOLO format" and "class names" in iss[0].detail


def test_pairs_dialog_labelme_fixes(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    root = FX.labelme(tmp_path / "lm", n=4)
    os.makedirs(os.path.join(root, "moved"))
    os.replace(os.path.join(root, "cam_frame_0001.jpg"), os.path.join(root, "moved", "cam_frame_0001.jpg"))
    _img(os.path.join(root, "new_0001.jpg"))                                           # not annotated yet
    w = _open(shell, qtbot, root)
    n0 = len(w.session.samples)
    w.qc_actions["pairs"].trigger()
    qtbot.waitUntil(lambda: "pairs" in w.open_dialogs, timeout=3000)
    d = w.open_dialogs["pairs"]
    qtbot.waitUntil(lambda: len(d.visible_rows()) == 2, timeout=10000)
    assert sorted(r.payload.kind for r in d.visible_rows()) == ["image_no_json", "json_no_image"]
    monkeypatch.setattr(d, "report", lambda *a, **k: None)
    for r in d.visible_rows():
        d.set_marked(r, True)
    d.apply_fixes()
    data = json.load(open(os.path.join(root, "cam_frame_0001.json")))
    assert data["imagePath"] == "moved/cam_frame_0001.jpg"
    new = json.load(open(os.path.join(root, "new_0001.json")))
    assert new["shapes"] == [] and new["imagePath"] == "new_0001.jpg" and new["imageWidth"] == 32
    assert len(w.session.samples) == n0 + 1                                             # the new JSON is a sample
    qtbot.waitUntil(lambda: d.worker is None or not d.worker.isRunning(), timeout=10000)
    qtbot.waitUntil(lambda: len(d.visible_rows()) == 0, timeout=5000)
