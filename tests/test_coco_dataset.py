# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""COCO JSON opened in place: layouts found (annotations/instances_<split>.json, Roboflow's _annotations.coco.json),
shapes read (boxes, polygons in pieces, RLE masks as outlines), saved losslessly (untouched objects keep every
field, a relabel changes only the category, a new label becomes a category), one write per batch, per-image Undo,
taking images out and putting them back, renaming an image — and the window: open, edit on the image, Ctrl+Z."""
import json
import shutil
from pathlib import Path

from . import fixtures as FX
from .test_aio_image import shell  # noqa: F401


def _coco(tmp: Path) -> Path:
    from AIO_QC.core import dataset as D, export as EX
    lm = Path(FX.labelme(tmp / "lm", n=6))
    a = D.open_dataset(lm, D.DatasetType.LABELME)
    smp = a.discover()
    labels = sorted({x["label"] for s in smp for x in a.load(s).shapes})
    out = tmp / "coco"
    EX.export({"train": smp[:4], "val": smp[4:]}, a, lm, out, "coco", labels)
    # extra fields that must survive a save, a piece-wise object and an RLE mask
    p = out / "annotations" / "instances_train.json"
    d = json.loads(p.read_text())
    d["annotations"][0]["attributes"] = {"occluded": True}
    im0 = d["images"][0]["id"]
    d["annotations"].append({"id": 9001, "image_id": im0, "category_id": 1, "iscrowd": 0, "area": 50,
                             "bbox": [5, 5, 40, 40], "segmentation": [[5, 5, 20, 5, 20, 20], [30, 30, 45, 30, 45, 45]]})
    d["annotations"].append({"id": 9002, "image_id": im0, "category_id": 1, "iscrowd": 1, "area": 100, "bbox":
                             [100, 100, 10, 10], "segmentation": {"size": [240, 320], "counts":
                                                                  [100 * 240 + 100] + [10, 230] * 9 + [10]}})
    p.write_text(json.dumps(d))
    return out


def test_layouts_and_reading(tmp_path):
    from AIO_QC.core import dataset as D
    from AIO_QC.core.coco_dataset import CocoAdapter, find_coco_files
    root = _coco(tmp_path)
    assert D.detect_dataset_types(root)[0] == D.DatasetType.COCO
    a = CocoAdapter(root)
    smp = a.discover()
    assert len(smp) == 6 and {s.split for s in smp} == {"train", "val"}
    assert all(Path(s.image_path).is_file() for s in smp)
    ann = a.load(smp[0])
    kinds = [(sh.get("_ann_id"), sh["shape_type"]) for sh in ann.shapes]
    assert (9001, "polygon") in kinds and kinds.count((9001, "polygon")) == 2       # an object in two pieces
    assert any(sh.get("_rle") for sh in ann.shapes)                                # the RLE mask, traced
    # Roboflow: train/_annotations.coco.json with the images next to it
    rf = tmp_path / "rf" / "train"
    rf.mkdir(parents=True)
    for s in smp[:2]:
        shutil.copy(s.image_path, rf / Path(s.image_path).name)
    (rf / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": Path(smp[0].image_path).name, "width": 320, "height": 240}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "thing"}]}))
    assert find_coco_files(tmp_path / "rf") and D.detect_dataset_types(tmp_path / "rf") == [D.DatasetType.COCO]
    b = CocoAdapter(tmp_path / "rf")
    s2 = b.discover()
    assert len(s2) == 1 and s2[0].split == "train" and b.load(s2[0]).shapes[0]["label"] == "thing"


def test_saving_is_lossless(tmp_path):
    from AIO_QC.core.coco_dataset import CocoAdapter
    root = _coco(tmp_path)
    p = root / "annotations" / "instances_train.json"
    a = CocoAdapter(root)
    smp = a.discover()
    first = smp[0]
    before = json.loads(p.read_text())
    ann = a.load(first)
    # relabel one object, move another, add one, delete the RLE one
    box = next(sh for sh in ann.shapes if sh["_ann_id"] not in (9001, 9002))
    box["label"] = "brand_new"
    ann.shapes = [sh for sh in ann.shapes if not sh.get("_rle")]
    ann.shapes.append({"label": "car", "points": [[1, 1], [9, 9]], "shape_type": "rectangle", "_geom_dirty": True})
    a.save(first, ann)
    d = json.loads(p.read_text())
    cats = {c["name"]: c["id"] for c in d["categories"]}
    assert "brand_new" in cats
    by_id = {x["id"]: x for x in d["annotations"]}
    assert 9002 not in by_id and by_id[9001]["segmentation"] == before["annotations"][-2]["segmentation"]
    relabelled = by_id[box["_ann_id"]]
    assert relabelled["category_id"] == cats["brand_new"]
    if box["_ann_id"] == before["annotations"][0]["id"]:
        assert relabelled["attributes"] == {"occluded": True}                      # kept as written
    new = [x for x in d["annotations"] if x["bbox"] == [1, 1, 8, 8]]
    assert len(new) == 1 and new[0]["id"] > 9002
    others = [x for x in before["annotations"] if x["image_id"] != before["images"][0]["id"]]
    assert all(by_id[x["id"]] == x for x in others)                               # other images untouched
    # per-image undo
    u = a.unit_bytes(first)
    a.write_unit(first, json.dumps({"anns": [x for x in before["annotations"] if x["image_id"] ==
                                             before["images"][0]["id"]]}).encode())
    assert {x["id"] for x in json.loads(p.read_text())["annotations"]} >= {9001, 9002}
    a.write_unit(first, u)
    # one write for many saves
    import os
    m0 = os.stat(p).st_mtime_ns
    with a.batch():
        for s in smp[:4]:
            x = a.load(s)
            a.save(s, x)
        assert os.stat(p).st_mtime_ns == m0
    # out of the file and back; an image renamed
    rec = a.remove_samples([smp[1]])
    assert len(json.loads(p.read_text())["images"]) == 3
    a.put_back(rec)
    assert len(json.loads(p.read_text())["images"]) == 4
    newp = Path(smp[2].image_path).with_name("renamed.jpg")
    shutil.move(smp[2].image_path, newp)
    ns = a.relocate(smp[2], newp)
    assert ns.key == str(newp) and any(i["file_name"].endswith("renamed.jpg") for i in json.loads(p.read_text())
                                       ["images"])


def test_window_edit_undo_trash(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    root = _coco(tmp_path)
    p = root / "annotations" / "instances_train.json"
    w = shell.window("image")
    w.open_dataset(root, D.COCO)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    s = w.session
    assert s.shared_ann and len(s.samples) == 6 and sum(s.label_totals().values()) > 0
    k0, k1 = s.samples[0].key, s.samples[1].key
    w.file_dock.select_key(k0)
    qtbot.waitUntil(lambda: w.current_key == k0, timeout=5000)
    n0 = len(s.load(k0).shapes)
    assert w.canvas.editable
    assert w.inline.commit(lambda ann: (ann.shapes.pop(0), "deleted one")[1])
    assert len(s.load(k0).shapes) == n0 - 1
    # an edit on another image, then Undo of the first one still works (per image)
    w.file_dock.select_key(k1)
    qtbot.waitUntil(lambda: w.current_key == k1, timeout=5000)
    n1 = len(s.load(k1).shapes)
    assert w.inline.commit(lambda ann: (ann.shapes.pop(0), "deleted one")[1])
    w.inline.undo()
    assert len(s.load(k1).shapes) == n1
    w.inline.undo()
    assert len(s.load(k0).shapes) == n0
    # take an image out of the file, and put it back
    got = []
    s.trashed.connect(got.append)
    done, errors = s.trash_samples([k1])
    assert done == [k1] and not errors and len(json.loads(p.read_text())["images"]) == 3 and k1 not in s.by_key
    back, _left = s.untrash(got[0])
    assert back == 1 and len(json.loads(p.read_text())["images"]) == 4 and k1 in s.by_key
