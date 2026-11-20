# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""More formats. Convert & Split writes Pascal VOC, CVAT 1.1, YOLO OBB, DOTA and YOLO pose (VOC and CVAT read back
to the same boxes); Import reads KITTI, VGG VIA, Supervisely (bitmaps too), CreateML, Open Images, Datumaro, Label
Studio brush masks and COCO keypoints — each found by the detector, images matched, written as LabelMe."""
import base64
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from . import fixtures as FX


def _export(tmp, fmt, n=4):
    from AIO_QC.core import dataset as D, export as EX
    lm = Path(FX.labelme(tmp / "lm", n=n))
    a = D.open_dataset(lm, D.DatasetType.LABELME)
    smp = a.discover()
    labels = ["car", "plate", "wheel"]
    out = tmp / fmt
    st = EX.export({"train": smp[:3], "val": smp[3:]}, a, lm, out, fmt, labels)
    return lm, out, st


def test_voc_and_cvat_export_round_trip(tmp_path):
    from AIO_QC.core.prep import importers as IM
    _lm, out, _st = _export(tmp_path, "voc")
    xmls = sorted((out / "train" / "Annotations").glob("*.xml"))
    assert len(xmls) == 3 and (out / "labels.txt").read_text().split() == ["car", "plate", "wheel"]
    assert (out / "train" / "ImageSets" / "Main" / "train.txt").read_text().split()[0] == xmls[0].stem
    assert IM.detect(out / "train" / "Annotations") == "voc"
    recs, _o = IM.read_voc(out / "train" / "Annotations")
    car = next(o for o in recs[0].objs if o.label == "car")
    x = 40
    assert car.points == [[x, 40], [x + 100, 120]]
    # CVAT
    _lm, out2, _st2 = _export(tmp_path, "cvat")
    p = out2 / "annotations.xml"
    assert IM.detect(p) == "cvat"
    recs, order = IM.read_cvat(p)
    kinds = sorted({o.kind for r in recs for o in r.objs})
    assert set(order) >= {"car", "plate", "wheel"} and {"rectangle", "polygon"} <= set(kinds)
    assert len(recs) == 4 and recs[0].ref.startswith("train/images/")
    P = IM.plan(p, out2, tmp_path / "back", "auto", "labelme")
    assert not IM.run(P)["errors"]


def test_obb_dota_and_pose(tmp_path):
    _lm, out, _st = _export(tmp_path, "yolo_obb")
    line = (out / "train" / "labels" / "cam_frame_0000.txt").read_text().splitlines()
    assert len(line) == 3 and all(len(x.split()) == 9 for x in line)
    assert "kpt_shape" not in (out / "dataset.yaml").read_text() and "train: train/images" in \
        (out / "dataset.yaml").read_text()
    _lm, out, _st = _export(tmp_path, "dota")
    t = (out / "train" / "labelTxt" / "cam_frame_0000.txt").read_text().splitlines()
    assert t[0].split()[-2:] == ["car", "0"] and len(t[0].split()) == 10
    # pose: a person box with grouped keypoints; an ungrouped point on an image with two boxes is reported
    from AIO_QC.core import dataset as D, export as EX
    from AIO_QC.core.export_more import flip_idx
    root = tmp_path / "pose"
    root.mkdir()
    for i in range(2):
        FX._img(str(root / f"p{i}.jpg"), seed=i)
        shapes = [{"label": "person", "points": [[10, 10], [110, 210]], "shape_type": "rectangle", "group_id": 1},
                  {"label": "left_eye", "points": [[40, 30]], "shape_type": "point", "group_id": 1},
                  {"label": "right_eye", "points": [[60, 30]], "shape_type": "point", "group_id": 1,
                   "flags": {"occluded": True}},
                  {"label": "nose", "points": [[50, 40]], "shape_type": "point", "group_id": 1}]
        if i == 1:
            shapes += [{"label": "person", "points": [[200, 10], [300, 200]], "shape_type": "rectangle"},
                       {"label": "nose", "points": [[250, 40]], "shape_type": "point"}]
        json.dump({"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": f"p{i}.jpg", "imageData": None,
                   "imageHeight": 240, "imageWidth": 320}, open(root / f"p{i}.json", "w"))
    a = D.open_dataset(root, D.DatasetType.LABELME)
    smp = a.discover()
    st = EX.export({"train": smp}, a, root, tmp_path / "pose_out", "yolo_pose",
                   ["person", "left_eye", "right_eye", "nose"])
    y = (tmp_path / "pose_out" / "dataset.yaml").read_text()
    assert "kpt_shape: [3, 3]" in y and "flip_idx: [1, 0, 2]" in y and "nc: 1" in y
    rows = (tmp_path / "pose_out" / "train" / "labels" / "p0.txt").read_text().split()
    assert len(rows) == 5 + 9 and rows[0] == "0" and rows[7] == "2" and rows[10] == "1"
    assert st["dropped"]["points not tied to a box (give them the box's group id)"] == 1
    assert flip_idx(["a", "b"]) is None


def _imgs(d: Path, names):
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        FX._img(str(d / n), seed=i)
    return d


def _run(tmp, src, imgs, fmt, out="o"):
    from AIO_QC.core.prep import importers as IM
    assert IM.detect(src) == fmt, IM.detect(src)
    P = IM.plan(src, imgs, tmp / out, "auto", "labelme")
    st = IM.run(P)
    assert not st["errors"] and st["done"] >= 1
    return P, {p.stem: json.loads(p.read_text()) for p in (tmp / out).rglob("*.json")}


def test_kitti_via_createml_openimages(tmp_path):
    imgs = _imgs(tmp_path / "images", ["000001.png", "000002.png", "a.jpg"])
    k = tmp_path / "label_2"
    k.mkdir()
    (k / "000001.txt").write_text("Car 0.00 0 -1.58 10.0 20.0 110.0 90.0 1.6 1.6 3.9 1.0 1.5 10.0 -1.5\n"
                                  "DontCare -1 -1 -10 0 0 5 5 -1 -1 -1 -1000 -1000 -1000 -10\n"
                                  "Pedestrian 0.80 2 0.2 150 30 170 100 1.7 0.6 0.8 2 1.5 9 0.1\n")
    (k / "000002.txt").write_text("Cyclist 0.00 0 0 1 2 30 40 1 1 1 1 1 1 0\n")
    P, out = _run(tmp_path, k, imgs, "kitti")
    labs = [s["label"] for s in out["000001"]["shapes"]]
    assert labs == ["Car", "Pedestrian"] and out["000001"]["shapes"][1]["flags"] == {"truncated": True,
                                                                                     "occluded": True}
    # VGG VIA 2: the class in a region attribute (chosen), a file attribute as a whole-image class
    via = {"a.jpg123": {"filename": "a.jpg", "size": 123, "file_attributes": {"scene": "day"}, "regions": [
        {"shape_attributes": {"name": "rect", "x": 5, "y": 6, "width": 50, "height": 40},
         "region_attributes": {"note": "x", "type": "dog"}},
        {"shape_attributes": {"name": "polygon", "all_points_x": [1, 50, 30], "all_points_y": [1, 2, 60]},
         "region_attributes": {"type": {"cat": True, "dog": False}}},
        {"shape_attributes": {"name": "circle", "cx": 100, "cy": 100, "r": 10}, "region_attributes": {"type": "ball"}},
        {"shape_attributes": {"name": "ellipse", "cx": 200, "cy": 100, "rx": 20, "ry": 10, "theta": 0.3},
         "region_attributes": {"type": "egg"}}]}}
    (tmp_path / "via.json").write_text(json.dumps({"_via_settings": {}, "_via_img_metadata": via}))
    from AIO_QC.core.prep import importers as IM
    from AIO_QC.core.prep.readers_more import via_attributes
    assert IM.detect(tmp_path / "via.json") == "via" and via_attributes(tmp_path / "via.json")[0] == "type"
    P = IM.plan(tmp_path / "via.json", imgs, tmp_path / "ov", "auto", "labelme", spec={"attr": "type"})
    IM.run(P)
    d = json.loads((tmp_path / "ov" / "a.json").read_text())
    assert [s["label"] for s in d["shapes"]] == ["dog", "cat", "ball", "egg"] and d["flags"] == {"day": True}
    assert [s["shape_type"] for s in d["shapes"]][:3] == ["rectangle", "polygon", "circle"]
    # CreateML: centre + size
    (tmp_path / "cml.json").write_text(json.dumps([{"image": "a.jpg", "annotations": [
        {"label": "box", "coordinates": {"x": 60, "y": 50, "width": 40, "height": 20}}]}]))
    _P, out = _run(tmp_path, tmp_path / "cml.json", imgs, "createml", "oc")
    assert out["a"]["shapes"][0]["points"] == [[40, 40], [80, 60]]
    # Open Images: 0…1 boxes, class ids → names from class-descriptions
    oi = tmp_path / "oi"
    oi.mkdir()
    (oi / "train-annotations-bbox.csv").write_text(
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
        "a,xclick,/m/01g317,1,0.25,0.5,0.5,1.0,1,0,0,0,0\n")
    (oi / "class-descriptions-boxable.csv").write_text("/m/01g317,Person\n")
    _P, out = _run(tmp_path, oi / "train-annotations-bbox.csv", imgs, "openimages", "oo")
    s = out["a"]["shapes"][0]
    assert s["label"] == "Person" and s["points"] == [[80, 120], [160, 240]] and s["flags"] == {"isoccluded": True}


def _ls_rle(mask_rgba: np.ndarray) -> list:
    """Label Studio's brush RLE, written with repeat runs only (what its decoder reads)."""
    arr = mask_rgba.reshape(-1)
    bits = f"{len(arr):032b}" + f"{7:05b}" + "".join(f"{s - 1:04b}" for s in (3, 4, 8, 16))
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[j] == arr[i] and j - i < 65536:
            j += 1
        n = j - i
        idx = next(k for k, s in enumerate((3, 4, 8, 16)) if n - 1 < 2 ** s)
        bits += "1" + f"{idx:02b}" + f"{n - 1:0{(3, 4, 8, 16)[idx]}b}" + f"{int(arr[i]):08b}"
        i = j
    bits += "0" * (-len(bits) % 8)
    return [int(bits[k:k + 8], 2) for k in range(0, len(bits), 8)]


def test_supervisely_datumaro_brush_keypoints(tmp_path):
    from AIO_QC.core.prep.readers_more import decode_ls_rle
    # Supervisely project: meta.json + ds/ann/*.json, images in ds/img/; a bitmap object traced to a polygon
    proj = tmp_path / "sly"
    _imgs(proj / "ds" / "img", ["s.jpg"])
    (proj / "meta.json").write_text(json.dumps({"classes": [{"title": "car"}, {"title": "road"}]}))
    m = np.zeros((40, 60, 4), np.uint8)
    m[5:30, 10:50] = 255
    ok, png = cv2.imencode(".png", m)
    bm = base64.b64encode(zlib.compress(png.tobytes())).decode()
    (proj / "ds" / "ann").mkdir()
    (proj / "ds" / "ann" / "s.jpg.json").write_text(json.dumps({
        "size": {"width": 320, "height": 240}, "tags": [{"name": "sunny"}], "objects": [
            {"classTitle": "car", "geometryType": "rectangle", "points": {"exterior": [[10, 20], [100, 90]]}},
            {"classTitle": "road", "geometryType": "bitmap", "bitmap": {"data": bm, "origin": [100, 100]}}]}))
    _P, out = _run(tmp_path, proj, proj, "supervisely")
    d = out["s"]
    assert [s["label"] for s in d["shapes"]] == ["car", "road"] and d["flags"] == {"sunny": True}
    xs = [p[0] for p in d["shapes"][1]["points"]]
    assert 108 <= min(xs) <= 112 and 146 <= max(xs) <= 150
    # Datumaro
    imgs = _imgs(tmp_path / "im", ["d1.jpg"])
    (tmp_path / "dm.json").write_text(json.dumps({
        "info": {}, "categories": {"label": {"labels": [{"name": "cat"}, {"name": "eye"}, {"name": "indoor"}]}},
        "items": [{"id": "d1", "image": {"path": "d1.jpg", "size": [240, 320]}, "annotations": [
            {"type": "bbox", "label_id": 0, "bbox": [10, 10, 50, 40], "group": 3},
            {"type": "points", "label_id": 1, "points": [20, 20, 30, 20], "group": 3},
            {"type": "polygon", "label_id": 0, "points": [100, 100, 150, 100, 150, 150]},
            {"type": "label", "label_id": 2}]}]}))
    _P, out = _run(tmp_path, tmp_path / "dm.json", imgs, "datumaro", "od")
    sh = out["d1"]["shapes"]
    assert [s["shape_type"] for s in sh] == ["rectangle", "point", "point", "polygon"]
    assert sh[0]["group_id"] == sh[1]["group_id"] == 3 and out["d1"]["flags"] == {"indoor": True}
    # Label Studio brush: the decoder, then an import
    mm = np.zeros((24, 32, 4), np.uint8)
    mm[4:20, 6:26, 3] = 255
    assert (decode_ls_rle(_ls_rle(mm), 24, 32) == (mm[:, :, 3] > 0)).all()
    big = np.zeros((240, 320, 4), np.uint8)
    big[50:150, 60:200] = 255
    _imgs(tmp_path / "ls", ["b.jpg"])
    (tmp_path / "ls.json").write_text(json.dumps([{"data": {"image": "/data/upload/1/0a1b2c3d-b.jpg"}, "annotations": [
        {"id": 1, "result": [{"id": "r1", "type": "brushlabels", "original_width": 320, "original_height": 240,
                              "value": {"format": "rle", "rle": _ls_rle(big), "brushlabels": ["blob"]}}]}]}]))
    _P, out = _run(tmp_path, tmp_path / "ls.json", tmp_path / "ls", "labelstudio", "ol")
    s = out["b"]["shapes"][0]
    xs, ys = [p[0] for p in s["points"]], [p[1] for p in s["points"]]
    assert s["label"] == "blob" and min(xs) <= 61 and max(xs) >= 198 and min(ys) <= 51 and max(ys) >= 148
    # COCO keypoints: one point per visible keypoint, grouped with the box
    _imgs(tmp_path / "ck", ["k.jpg"])
    (tmp_path / "kp.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "k.jpg", "width": 320, "height": 240}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye"]}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 200], "iscrowd": 0,
                         "area": 2e4, "keypoints": [50, 40, 2, 0, 0, 0, 60, 30, 1], "num_keypoints": 2}]}))
    _P, out = _run(tmp_path, tmp_path / "kp.json", tmp_path / "ck", "coco", "ok")
    sh = out["k"]["shapes"]
    pts = [s for s in sh if s["shape_type"] == "point"]
    assert [s["label"] for s in pts] == ["nose", "right_eye"]
    assert len({s["group_id"] for s in sh}) == 1 and sh[0]["group_id"] is not None
