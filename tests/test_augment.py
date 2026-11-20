# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Image Augmentation, the engine and the dataset writer: every effect, labels that follow the geometry, recipes
(clean / legacy import / on disk), the Ultralytics step, and Generate — parallel, repeatable, resumable, stoppable,
with pasted objects and the three ways of handling the originals."""
import json
import pickle
import threading

import cv2
import numpy as np
import pytest

from AIO_QC.core import augment as A
from AIO_QC.core import augment_ops as G

from . import fixtures as FX


def _img(w=320, h=240, seed=1):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 2)


def _only(key, **cfg):
    return G._with({key: dict(cfg, p=1.0)}, mode="each", name=key)


SHAPES = [{"label": "a", "shape_type": "rectangle", "points": [[40, 30], [140, 110]]},
          {"label": "b", "shape_type": "polygon", "points": [[150, 60], [250, 70], [240, 180], [160, 170]]},
          {"label": "c", "shape_type": "obb", "points": [[20, 150], [80, 160], [75, 200], [15, 190]]},
          {"label": "d", "shape_type": "circle", "points": [[160, 120], [170, 120]]},
          {"label": "e", "shape_type": "point", "points": [[10, 10]]},
          {"label": "f", "shape_type": "linestrip", "points": [[10, 10], [100, 50], [200, 30]]}]


def test_every_built_in_effect_runs_and_repeats():
    img = _img()
    keys = [o["key"] for o in G.CATALOG if not o["needs"]]
    assert len(keys) >= 45 and {"rotate", "elastic", "fog", "jpeg", "cutout", "motion_blur"} <= set(keys)
    for k in keys:
        a = G.Augmenter(_only(k))
        out, sh, log = a.apply(img, SHAPES, seed=3)
        assert out.shape == img.shape and out.dtype == np.uint8, k
        assert [x for x, _v in log] == [k] and not isinstance(log[0][1], str), (k, log)
        out2, sh2, _ = a.apply(img, SHAPES, seed=3)
        assert np.array_equal(out, out2) and sh == sh2, k            # the same seed: the same picture and labels
        if G.OPS[k]["kind"] in ("pixel",):
            assert sh == [dict(s) for s in SHAPES], k                # pixels only: labels untouched
    assert "·" in G.describe([("rotate", {"deg": 12.34}), ("jpeg", {"quality": 30})])
    # grey and RGBA images keep their layout
    for im in (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.dstack([img, np.full(img.shape[:2], 200, np.uint8)])):
        out, _s, _l = G.Augmenter(G.presets()["Strong"][0]).apply(im, [], seed=1)
        assert out.shape == im.shape


def test_labels_follow_the_geometry():
    img = _img()
    out, sh, _ = G.Augmenter(_only("fliplr")).apply(img, SHAPES, seed=0)
    assert np.array_equal(out, img[:, ::-1])
    W = 320 - 1
    assert sh[0]["points"] == [[W - 140, 30], [W - 40, 110]]                       # a rectangle stays a rectangle
    assert sh[1]["points"][0] == [W - 150, 60] and len(sh[1]["points"]) == 4
    assert len(sh[2]["points"]) == 4 and sh[3]["points"][0] == [W - 160, 120] and sh[4]["points"] == [[W - 10, 10]]
    assert [s["label"] for s in sh] == list("abcdef")
    # where a marked point goes under each warp (the label is where the pixels went)
    canvas = np.zeros((240, 320, 3), np.uint8)
    pts = [(80, 70), (220, 150), (160, 110)]
    for x, y in pts:
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
    marks = [{"shape_type": "point", "points": [list(p)]} for p in pts]
    for k in ("rotate", "shear", "scale", "perspective", "elastic", "piecewise", "lens"):
        for seed in range(3):
            out, sh, _ = G.Augmenter(_only(k)).apply(canvas, marks, seed=seed)
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            for s in sh:
                x, y = (int(round(v)) for v in s["points"][0])
                win = g[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
                assert win.size and win.max() > 60, (k, seed)
    # an object pushed out of the image: clipped, then dropped under the visible share
    r = _only("translate", frac=[0.45, 0.45])
    r["min_visible"] = 0.3
    out, sh, _ = G.Augmenter(r).apply(img, [SHAPES[1]], seed=0)                  # polygon x 150..250 → 294..394
    assert len(sh) in (0, 1) and all(max(p[0] for p in s["points"]) <= 320 for s in sh)
    r["min_visible"] = 0.9
    assert G.Augmenter(r).apply(img, [SHAPES[1]], seed=0)[1] == []
    # an oriented box stays 4 points, in the same order
    out, sh, _ = G.Augmenter(_only("rotate", deg=[20, 20])).apply(img, [SHAPES[2]], seed=0)
    assert len(sh[0]["points"]) == 4


def test_recipes_modes_and_legacy_import(tmp_path):
    img = _img()
    for mode, lo, hi in (("one", 0, 1), ("some", 2, 3), ("each", 0, 99)):
        r = G.presets()["Strong"][0]
        r.update(mode=mode, min=2, max=3)
        ns = {len(G.Augmenter(r).apply(img, [], seed=s)[2]) for s in range(12)}
        assert min(ns) >= lo and max(ns) <= hi, (mode, ns)
    weak, strong = _only("rotate", deg=[30, 30]), _only("rotate", deg=[30, 30])
    weak["strength"] = 0.5
    assert G.Augmenter(weak).apply(img, [], seed=1)[2][0][1]["deg"] == pytest.approx(15.0)
    assert G.Augmenter(strong).apply(img, [], seed=1)[2][0][1]["deg"] == pytest.approx(30.0)
    junk = G.clean({"mode": "?", "min": 99, "ops": {"rotate": {"on": 1, "p": 7, "deg": [500, -500]}, "nope": {}}})
    assert junk["mode"] == "some" and junk["ops"]["rotate"]["deg"] == [-180, 180] and junk["ops"]["rotate"]["p"] == 1.0
    assert "nope" not in junk["ops"] and set(junk["ops"]) == set(G.OPS)
    # the older imgaug pipeline's aug_parameters.yaml
    y = tmp_path / "aug_parameters.yaml"
    y.write_text("apply_geo_transformation:\n  enabled: True\n  rotate:\n    range: [-30, 30]\n    apply: True\n"
                 "    prob: 0.5\n  fliplr:\n    apply: False\n    prob: 0.3\n"
                 "apply_color_transformation:\n  enabled: True\n  hue:\n    range: [-20, 20]\n    apply: True\n"
                 "    prob: 1\napply_noise_artifact:\n  enabled: False\n  gausBlur:\n    scale: [0.5, 5.0]\n"
                 "    apply: True\n    prob: 1\n"
                 "apply_lightning_effects:\n  enabled: True\n  motionBlur:\n    kernel: [8, 15]\n"
                 "    angle: [150, 360]\n    apply: True\n    prob: 0.4\n")
    r = A.import_file(y)
    on = {k for k, v in r["ops"].items() if v["on"]}
    assert on == {"rotate", "hue", "motion_blur"} and r["mode"] == "one" and r["name"] == "aug_parameters"
    assert r["ops"]["rotate"]["deg"] == [-30, 30] and r["ops"]["rotate"]["p"] == 0.5
    assert r["ops"]["motion_blur"]["kernel"] == [8, 15]
    assert len([k for k, v in G.from_legacy({})["ops"].items() if v["on"]]) > 20      # its stock settings
    # on disk
    r["name"] = "My plates / test"
    p = A.save_recipe(r)
    assert p.is_file() and "My plates / test" in A.recipe_names() and A.all_names()[0] == "My plates / test"
    assert A.load_recipe("My plates / test")["ops"]["hue"]["deg"] == [-20, 20]
    assert A.load_recipe("Balanced")["name"] == "Balanced" and A.load_recipe("no such") is None
    assert A.import_file(p)["name"] == "My plates / test"
    assert A.delete_recipe("My plates / test") and "My plates / test" not in A.recipe_names()
    with pytest.raises(ValueError):
        (tmp_path / "x.json").write_text('{"a": 1}')
        A.import_file(tmp_path / "x.json")


def test_libraries_are_optional():
    info = G.libraries()
    assert set(info) == {"albumentations", "imgaug"}
    for name, d in info.items():
        keys = [o["key"] for o in G.CATALOG if o["needs"] == name]
        assert set(d["effects"]) | set(d["broken"]) <= set(keys)
        if d["version"] is None:                                       # not installed here: never offered, never run
            assert d["error"] and not any(G.available(k) for k in keys)
            r = _only(keys[0])
            a = G.Augmenter(r)
            assert a.skipped == [keys[0]] and a.active == [] and a.apply(_img(), [], seed=0)[2] == []
        else:
            for k in d["effects"]:
                out = G.Augmenter(_only(k)).apply(_img(), [], seed=0)[0]
                assert out.shape == (240, 320, 3) and out.dtype == np.uint8
    G._numpy2_shim()                                                   # harmless, whatever the NumPy
    assert hasattr(np, "sctypes") and np.prod([2, 3]) == 6


class _Instances:
    """What the Ultralytics step needs of ultralytics.utils.instance.Instances."""

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xyxy", normalized=True):
        self.bboxes, self.segments, self.keypoints = np.asarray(bboxes, np.float32), segments, keypoints
        self.format, self.normalized = bbox_format, normalized

    def __len__(self):
        return len(self.bboxes)

    def convert_bbox(self, format):
        assert self.format == format == "xyxy"

    def denormalize(self, w, h):
        if self.normalized:
            self.bboxes = self.bboxes * np.array([w, h, w, h], np.float32)
            if self.segments is not None and len(self.segments):
                self.segments = self.segments * np.array([w, h], np.float32)
            self.normalized = False


def test_the_ultralytics_step():
    img = _img()
    t = pickle.loads(pickle.dumps(G.UltralyticsTransform(_only("fliplr"))))      # as a worker process gets it
    seg = np.array([[[0.25, 0.25], [0.5, 0.25], [0.5, 0.5], [0.25, 0.5]]], np.float32)
    lab = {"img": img, "cls": np.array([[3.0]]), "instances": _Instances([[0.25, 0.25, 0.5, 0.5]], seg.copy())}
    out = t(lab)
    assert np.array_equal(out["img"], img[:, ::-1])
    b = out["instances"].bboxes[0]
    assert out["instances"].normalized is False and b.tolist() == pytest.approx([159, 60, 239, 120], abs=1.0)
    assert out["instances"].segments.shape == (1, 4, 2) and out["cls"].tolist() == [[3.0]]
    # boxes only; an object pushed out loses its row together with its class
    r = _only("translate", frac=[0.49, 0.49])
    r["min_visible"] = 0.5                               # (the other axis shifts at random: a full-height box
    lab = {"img": img, "cls": np.array([[0.0], [1.0]]),  # keeps over half of itself whatever that is)
           "instances": _Instances([[0.6, 0.1, 0.9, 0.9], [0.05, 0.0, 0.2, 1.0]], np.zeros((0, 4, 2), np.float32))}
    out = G.UltralyticsTransform(r)(lab)
    assert len(out["instances"]) == 1 and out["cls"].tolist() == [[1.0]]
    # pose: keypoints cannot follow yet — pixel effects only
    r = G._with({"fliplr": {"p": 1.0}, "invert": {"p": 1.0}}, mode="each")
    lab = {"img": img, "cls": np.array([[0.0]]),
           "instances": _Instances([[0.2, 0.2, 0.4, 0.4]], None, np.zeros((1, 17, 3), np.float32))}
    out = G.UltralyticsTransform(r)(lab)
    assert np.array_equal(out["img"], 255 - img) and out["instances"].bboxes[0].tolist() == pytest.approx([.2, .2, .4, .4])
    # a part of the images only; a failing image goes through unchanged
    t = G.UltralyticsTransform(_only("invert"), share=0.5)
    n = sum(not np.array_equal(t({"img": img, "instances": None})["img"], img) for _ in range(200))
    assert 60 < n < 140
    assert G.UltralyticsTransform(_only("invert"))({"img": None})["img"] is None


def _yolo_items(root):
    out = []
    for p in sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root).as_posix()
        lab = (root / rel.replace("images", "labels", 1)).with_suffix(".txt")
        out.append(A.Item(lab.relative_to(root).as_posix() if lab.is_file() else "", rel, rel.split("/")[0], rel))
    return out


def test_generate_yolo_parallel_repeatable_resumable(tmp_path):
    y = FX.yolo(tmp_path / "y", n=16)
    items = _yolo_items(y)
    copies = A.copies_per_item(items, 2, splits={"train"})
    assert sum(copies) == 2 * sum(1 for it in items if it.split == "train")
    recipe = G.presets()["Strong"][0]
    seen = []
    rep = A.run(A.Job(str(y), str(tmp_path / "a"), "yolo", recipe, seed=4, workers=2), items, copies,
                progress=lambda d, t, r: seen.append((d, t)))
    assert rep["written"] == sum(copies) and not rep["errors"] and not rep["cancelled"] and seen[-1][0] == seen[-1][1]
    assert rep["originals"] == 2 * len(items)                         # images + labels, linked
    new = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*__aug*"))
    assert len(new) == 2 * sum(copies) and not list((tmp_path / "a").rglob("*.tmp"))
    assert not list((tmp_path / "a" / "val").rglob("*__aug*"))        # val stays as it is
    for p in (tmp_path / "a").rglob("*__aug1.txt"):
        for ln in p.read_text().splitlines():
            vals = [float(v) for v in ln.split()[1:]]
            assert len(vals) == 4 and all(0 <= v <= 1 for v in vals)
    meta = json.loads((tmp_path / "a" / ".aio_qc" / "augmentation.json").read_text())
    assert meta["recipe"]["name"] == "Strong" and meta["report"]["written"] == sum(copies)
    assert "path:" in (tmp_path / "a" / "data.yaml").read_text()
    # one worker, the same seed: byte for byte the same files
    A.run(A.Job(str(y), str(tmp_path / "b"), "yolo", recipe, seed=4, workers=1), items, copies)
    assert all((tmp_path / "b" / f).read_bytes() == (tmp_path / "a" / f).read_bytes() for f in new)
    A.run(A.Job(str(y), str(tmp_path / "c"), "yolo", recipe, seed=5, workers=1), items, copies)
    assert any((tmp_path / "c" / f).read_bytes() != (tmp_path / "a" / f).read_bytes() for f in new)
    # again: nothing is made twice
    rep = A.run(A.Job(str(y), str(tmp_path / "a"), "yolo", recipe, seed=4, workers=2), items, copies)
    assert rep["written"] == 0 and rep["skipped"] == sum(copies)
    # stop in the middle, then go on
    ev = threading.Event()
    job = A.Job(str(y), str(tmp_path / "d"), "yolo", recipe, seed=4, workers=2)
    many = A.copies_per_item(items, 12)
    rep = A.run(job, items, many, progress=lambda d, t, r: ev.set() if d >= 4 else None, cancel=ev)
    assert rep["cancelled"] and 0 < rep["written"] < sum(many) and not list((tmp_path / "d").rglob("*.tmp"))
    rep2 = A.run(job, items, many)
    assert not rep2["cancelled"] and rep2["skipped"] >= rep["written"] and rep2["written"] + rep2["skipped"] == sum(many)


def test_generate_originals_three_ways_and_paste(tmp_path):
    import yaml
    y = FX.yolo(tmp_path / "y", n=10)
    items = _yolo_items(y)
    copies = A.copies_per_item(items, 1, splits={"train"})
    light = G.presets()["Light"][0]
    rep = A.run(A.Job(str(y), str(tmp_path / "ref"), "yolo", light, workers=1, originals=False, reference=True,
                      paste={"p": 1.0, "boxes": True, "count": [2, 2]}), items, copies)
    assert rep["originals"] == 0 and rep["pasted"] > 0 and rep["shapes_in"] > 2 * rep["written"]
    d = yaml.safe_load((tmp_path / "ref" / "data.yaml").read_text())
    assert d["train"] == [str(y / "train" / "images"), str(tmp_path / "ref" / "train" / "images")]
    assert d["val"] == str(y / "val" / "images") and d["names"] == {0: "car", 1: "plate"}
    assert not (tmp_path / "ref" / "val").exists() and not list((tmp_path / "ref").rglob("train_0.jpg"))
    A.run(A.Job(str(y), str(tmp_path / "none"), "yolo", light, workers=1, originals=False), items, copies)
    d = yaml.safe_load((tmp_path / "none" / "data.yaml").read_text())
    assert d["train"] == str(tmp_path / "none" / "train" / "images") and d["val"] == str(y / "val" / "images")
    # LabelMe: every shape type written back, pasted polygons added, the image next to its JSON
    lm = FX.labelme(tmp_path / "lm", n=8)
    its = [A.Item(p.relative_to(lm).as_posix(), "", "", p.stem) for p in sorted(lm.rglob("*.json"))]
    rep = A.run(A.Job(str(lm), str(tmp_path / "lm_aug"), "labelme", _only("fliplr"), workers=2, originals=False,
                      paste={"p": 1.0, "count": [2, 2], "scale": [0.5, 0.8], "classes": ["plate"]}),
                its, A.copies_per_item(its, 1))
    assert rep["written"] == 8 and rep["pasted"] >= 8 and not rep["errors"]
    j = sorted((tmp_path / "lm_aug").rglob("*__aug1.json"))[0]
    d = json.loads(j.read_text())
    src = json.loads((lm / its[0].ann).read_text())
    assert (j.parent / d["imagePath"]).is_file() and d["imageData"] is None and d["imageWidth"] == 320
    labels = [s["label"] for s in d["shapes"]]
    assert labels[:len(src["shapes"])] == [s["label"] for s in src["shapes"]]
    assert labels[len(src["shapes"]):] and set(labels[len(src["shapes"]):]) == {"plate"}
    assert not (tmp_path / "lm_aug" / its[0].ann).exists()            # originals left out
    # more copies for images with a rare class
    c = A.copies_per_item(its, 1, class_keys={"car": {i.key for i in its}, "rare": {its[0].key}}, balance=True, cap=5)
    assert c[0] == 5 and set(c[1:]) == {1}
    # class folders: images only
    cls = tmp_path / "cls"
    for name in ("cat", "dog"):
        (cls / name).mkdir(parents=True)
        cv2.imwrite(str(cls / name / "1.png"), _img(64, 48))
    its = [A.Item("", f"{n}/1.png", "", n) for n in ("cat", "dog")]
    rep = A.run(A.Job(str(cls), str(tmp_path / "cls_aug"), "images", light, workers=1), its, [2, 2])
    assert rep["written"] == 4 and (tmp_path / "cls_aug" / "cat" / "1__aug2.png").is_file()
