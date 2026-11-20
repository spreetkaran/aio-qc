# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Data Prep without the window: importers (COCO / VOC / CVAT / Label Studio → LabelMe and YOLO), Combine & Flatten,
the small tools (unzip, orphans, integrity, exact duplicates, resize), EXIF rotation and Pick-to-label — each one
planned, run, and undone from its log, on small folders made here."""
import base64
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from AIO_QC.core.prep import common as C
from AIO_QC.core.prep import exif_fix, flatten, importers as I, pick, tasks as T


def _img(path: Path, w=64, h=48, colour=(40, 120, 200), seed=0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((h, w, 3), colour, np.uint8)
    a[seed % h, :] = 255 - seed % 200                         # different bytes per seed
    cv2.imwrite(str(path), a)
    return path


def _lm(path: Path, image_path, shapes=None, data=None, w=64, h=48) -> Path:
    shapes = shapes if shapes is not None else [
        {"label": "car", "shape_type": "rectangle", "points": [[2, 3], [20, 30]], "flags": {}, "group_id": None,
         "description": ""}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": image_path,
                                "imageData": data, "imageHeight": h, "imageWidth": w}))
    return path


def _files(root: Path):
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
                  if p.is_file() and C.LOG_DIR not in p.parts)


# ============================================================================================================ common
def test_common_helpers(tmp_path):
    assert C.human(512) and C.human(5 * 1024 ** 3).endswith("GB")
    assert C.inside(tmp_path / "a" / "b", tmp_path) and not C.inside(tmp_path, tmp_path / "a")
    assert C.stem("x/y/photo.JPG") == "photo" and C.ext("x/photo.JPG").lower() == ".jpg"
    raw = _img(tmp_path / "a.png").read_bytes()
    assert C.sniff(raw[:32]) == ".png"
    b64 = base64.b64encode(raw).decode()
    assert C.peek_image_data(b64) == ".png" and C.b64_size(b64) == len(raw)
    assert C.decode_image_data("data:image/png;base64," + b64)[0] == raw
    with pytest.raises(ValueError):
        C.decode_image_data("bm90IGFuIGltYWdl")
    assert C.verify_bytes(raw)[0] and not C.verify_bytes(b"\x89PNG\r\n\x1a\nbroken")[0]
    ok, _why, wh = C.verify_file(tmp_path / "a.png")
    assert ok and tuple(wh) == (64, 48)
    assert C.file_sha(tmp_path / "a.png") == C.file_sha(tmp_path / "a.png") and C.file_sha(tmp_path / "nope") is None
    d1 = C.label_digest({"shapes": [{"label": "a", "points": [[1, 2]]}]})
    assert d1 == C.label_digest({"shapes": [{"label": "a", "points": [[1, 2]]}]})
    names = C.Names(tmp_path)
    assert names.get("a") != names.get("a")
    assert C.unique_path(tmp_path / "a.png") != tmp_path / "a.png"
    (tmp_path / "e1" / "e2").mkdir(parents=True)
    assert C.remove_empty_dirs([tmp_path / "e1" / "e2"], tmp_path) == 2


def test_runlog_records_and_undo_never_overwrites(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("A")
    b = tmp_path / "b.txt"
    b.write_text("B")
    log = C.RunLog(tmp_path / C.LOG_DIR, "t")
    moved = tmp_path / "moved" / "a.txt"
    moved.parent.mkdir()
    a.rename(moved)
    log.moved(a, moved)
    made = tmp_path / "new.txt"
    made.write_text("new")
    log.created(made)
    orig = log.keep_original(b)
    b.write_text("B2")
    log.rewritten(b, orig)
    edited = tmp_path / "edited.txt"
    edited.write_text("x")
    log.created(edited)
    edited.write_text("changed later")
    log.error("x", "s", "d", "boom")
    log.close()
    assert C.list_runs(tmp_path) == [log.jsonl_path]
    st = C.undo_run(log.jsonl_path)
    assert a.read_text() == "A" and not made.exists() and b.read_text() == "B"
    assert edited.exists() and any("changed after the run" in s for s in st["skipped"])
    assert not log.jsonl_path.exists() and log.jsonl_path.with_suffix(".undone.jsonl").exists()


# ============================================================================================================ importers
def _images(root: Path, names):
    for i, n in enumerate(names):
        _img(root / n, 100, 80, seed=i)
    return root


def _check_undo(out: Path, st):
    C.undo_run(st["undo"])
    assert _files(out) == []


def test_import_coco_every_kind_to_labelme_and_yolo(tmp_path):
    imgs = _images(tmp_path / "imgs", ["a.jpg", "sub/b.jpg", "c.jpg"])
    mask = np.zeros((80, 100), np.uint8)
    mask[10:40, 20:60] = 1
    counts, flat, val, run = [], mask.T.reshape(-1), 0, 0
    for v in flat:                                             # uncompressed (column-major) RLE
        if v != val:
            counts.append(run)
            run, val = 0, v
        run += 1
    counts.append(run)
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
                       {"id": 2, "file_name": "sub/b.jpg", "width": 100, "height": 80},
                       {"id": 3, "file_name": "missing.jpg", "width": 100, "height": 80},
                       {"id": 4, "file_name": "http://host/x/c.jpg"}],
            "categories": [{"id": 2, "name": "plate"}, {"id": 1, "name": "car"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 20]},
                            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5], "iscrowd": 1,
                             "segmentation": [[1, 1, 9, 1, 9, 9, 1, 9]]},
                            {"image_id": 2, "category_id": 2, "bbox": [20, 10, 40, 30],
                             "segmentation": {"size": [80, 100], "counts": counts}},
                            {"image_id": 2, "category_id": 1, "bbox": [1, 1, 3, 3], "keypoints": [1, 2, 2]},
                            {"image_id": 4, "category_id": 1, "bbox": [50, 50, 200, 200]},
                            {"image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]}]}
    src = tmp_path / "coco.json"
    src.write_text(json.dumps(coco))
    assert I.detect(src) == "coco"
    recs, order = I.read_coco(src)
    assert order == ["car", "plate", "kp0"] and sum(len(r.objs) for r in recs) == 6    # + an unnamed keypoint
    # the decoded mask became a polygon around the 40×30 block
    poly = next(o for r in recs if r.ref == "sub/b.jpg" for o in r.objs if o.kind == "polygon")
    xs, ys = [p[0] for p in poly.points], [p[1] for p in poly.points]
    assert (min(xs), min(ys), max(xs), max(ys)) == (20, 10, 59, 39)
    notes = []
    I.read_coco(src, use_masks=False, notes=notes)
    assert any("keypoint" in n for n in notes)
    with pytest.raises(ValueError, match="outside"):
        I.plan(src, imgs, imgs / "out")
    P = I.plan(src, imgs, tmp_path / "lm", output="labelme", link=False)
    rows = dict(P.rows)
    assert rows["Images found"] == 3 and rows["    not found (skipped)"] == 1
    assert "missing.jpg" in P.details["Images not found (skipped)"]
    st = I.run(P)
    assert st["done"] == 3 and not st["errors"] and st["copied"] == 3
    a = json.loads((tmp_path / "lm" / "a.json").read_text())
    assert [s["shape_type"] for s in a["shapes"]] == ["rectangle", "polygon"] and a["imagePath"] == "a.jpg"
    assert a["shapes"][1]["flags"] == {"iscrowd": True}
    c = json.loads((tmp_path / "lm" / "c.json").read_text())                 # size read from the image; clipped
    assert c["imageWidth"] == 100 and c["shapes"][0]["points"][1] == [100, 80]
    assert (tmp_path / "lm" / "sub" / "b.json").is_file()
    _check_undo(tmp_path / "lm", st)
    with pytest.raises(ValueError, match="new or empty"):
        (tmp_path / "busy").mkdir()
        (tmp_path / "busy" / "x").write_text("x")
        I.plan(src, imgs, tmp_path / "busy")
    for output in ("yolo_det", "yolo_seg"):
        out = tmp_path / output
        st = I.run(I.plan(src, imgs, out, output=output))
        assert st["done"] == 3 and (out / "data.yaml").is_file() and (out / "classes.txt").read_text() == "car\nplate\n"
        lines = (out / "labels" / "a.txt").read_text().split("\n")
        assert lines[0].split()[0] == "0" and lines[1].split()[0] == "1"
        assert len(lines[0].split()) == (5 if output == "yolo_det" else 9)
        assert (out / "images" / "sub" / "b.jpg").is_file()


def test_import_voc_cvat_and_labelstudio(tmp_path):
    imgs = _images(tmp_path / "imgs", ["v1.jpg", "d/v2.jpg", "f0.jpg", "f1.jpg", "photo.jpg"])
    voc = tmp_path / "voc"
    (voc / "d").mkdir(parents=True)
    (voc / "v1.xml").write_text("<annotation><filename>v1.jpg</filename><size><width>100</width><height>80</height>"
                                "</size><object><name>car</name><difficult>1</difficult><bndbox><xmin>1</xmin>"
                                "<ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object><object><name>"
                                "</name></object></annotation>")
    (voc / "d" / "v2.xml").write_text("<annotation><filename>v2.jpg</filename><object><name>bus</name><bndbox>"
                                      "<xmin>x</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>"
                                      "</annotation>")
    (voc / "bad.xml").write_text("<annotation><oops")
    assert I.detect(voc) == "voc" and I.detect(voc / "v1.xml") == "voc"
    notes = []
    recs, order = I.read_voc(voc, notes)
    assert order == ["bus", "car"] or order == ["car", "bus"]
    assert {r.ref.replace("\\", "/") for r in recs} == {"v1.jpg", "d/v2.jpg"} and any("bad.xml" in n for n in notes)
    assert next(r for r in recs if r.ref == "v1.jpg").objs[0].flags == {"difficult": True}
    st = I.run(I.plan(voc, imgs, tmp_path / "voc_out", link=True))
    assert st["done"] == 2 and st["linked"] + st["copied"] == 2

    cvat = tmp_path / "cvat.xml"
    cvat.write_text("""<annotations><meta><task><labels><label><name>car</name></label><label><name>text</name>
      </label></labels></task></meta>
      <image name="v1.jpg" width="100" height="80">
        <box label="car" xtl="1" ytl="2" xbr="30" ybr="40" occluded="1"><attribute name="parked">true</attribute>
          <attribute name="colour">red</attribute></box>
        <box label="car" xtl="10" ytl="10" xbr="30" ybr="20" rotation="30"/>
        <polygon label="text" points="1,1;9,1;9,9"><attribute name="text">AB12</attribute></polygon>
        <polyline label="lane" points="0,0;5,5"/>
        <points label="kp" points="3,4"/>
        <points label="kp" points="3,4;5,6"/>
        <ellipse label="wheel" cx="50" cy="40" rx="10" ry="10"/>
        <ellipse label="wheel" cx="50" cy="40" rx="20" ry="10" rotation="15"/>
        <mask label="blob" left="10" top="10" width="10" height="10" rle="22,6,4,6,4,6,4,6,4,6,32"/>
        <tag label="nolabelshape"/>
      </image>
      <track id="0" label="car"><box frame="0" xtl="1" ytl="1" xbr="9" ybr="9" outside="0"/>
        <box frame="1" xtl="2" ytl="2" xbr="9" ybr="9" outside="1"/></track>
    </annotations>""")
    assert I.detect(cvat) == "cvat"
    notes = []
    recs, order = I.read_cvat(cvat, tmp_path / "imgs", notes)
    kinds = [o.kind for o in recs[0].objs]
    assert kinds == ["rectangle", "polygon", "polygon", "linestrip", "point", "linestrip", "circle", "polygon",
                     "polygon"], kinds
    box = recs[0].objs[0]
    assert box.flags == {"parked": True, "occluded": True} and box.attrs == {"colour": "red"}
    assert recs[0].objs[2].description == "AB12" and "video tracks" in notes[0]
    assert {"lane", "kp", "wheel", "blob"} <= set(order)
    assert recs[-1].ref.replace("\\", "/") == "d/v2.jpg" or recs[-1].ref                # frame 0 = first image
    P = I.plan(cvat, imgs, tmp_path / "cvat_out", output="yolo_det")
    st = I.run(P)
    assert st["done"] >= 1 and st["skipped"] >= 2                        # points / lines have no box

    ls = tmp_path / "ls.json"
    res = lambda t, v, i="r1": {"id": i, "type": t, "value": v, "original_width": 100, "original_height": 80}  # noqa
    ls.write_text(json.dumps([
        {"data": {"image": "/data/upload/1/1a2b3c4d-photo.jpg"},
         "annotations": [{"id": 1, "updated_at": "2024", "result": [
             res("rectanglelabels", {"x": 10, "y": 10, "width": 20, "height": 25, "rectanglelabels": ["car"]}),
             res("textarea", {"text": ["AB", "12"]}),
             res("rectanglelabels", {"x": 10, "y": 10, "width": 20, "height": 25, "rotation": 30,
                                     "rectanglelabels": ["car"]}, "r2"),
             res("polygonlabels", {"points": [[1, 1], [50, 1], [50, 50]], "polygonlabels": ["zone"]}, "r3"),
             res("keypointlabels", {"x": 5, "y": 5, "keypointlabels": ["kp"]}, "r4"),
             res("ellipselabels", {"x": 50, "y": 50, "radiusX": 10, "radiusY": 5, "ellipselabels": ["e"]}, "r5"),
             res("brushlabels", {"brushlabels": ["m"]}, "r6"),
             res("choices", {"choices": ["day"]}, "r7"),
             {"id": "r8", "type": "rectanglelabels", "value": {"rectanglelabels": ["car"]}}]},
             {"id": 0, "was_cancelled": True, "result": []}]},
        {"data": {"img": "f0.jpg"}, "annotations": []}]))
    assert I.detect(ls) == "labelstudio"
    notes = []
    recs, order = I.read_labelstudio(ls, notes)
    assert len(recs) == 1 and [o.kind for o in recs[0].objs] == ["rectangle", "polygon", "polygon", "point", "polygon"]
    assert recs[0].objs[0].description == "AB\n12" and "brushlabels" in notes[0] and "no image size" in notes[0]
    assert I.ImageIndex(imgs).find(recs[0].ref) == (str(imgs / "photo.jpg"), "name")
    st = I.run(I.plan(ls, imgs, tmp_path / "ls_out"))
    assert st["done"] == 1 and json.loads((tmp_path / "ls_out" / "photo.json").read_text())["shapes"]
    with pytest.raises(ValueError):
        I.read_any("nope", ls, imgs)
    assert I.detect(tmp_path / "imgs") is None


def test_rle_compressed_string_round_trip():
    mask = np.zeros((6, 5), np.uint8)
    mask[1:4, 1:3] = 1
    # COCO's compressed form of the column-major runs → encoded as pycocotools does
    def enc(cnts):
        s, out = list(cnts), []
        for i, x in enumerate(s):
            if i > 2:
                x -= s[i - 2]
            more = True
            while more:
                c = x & 0x1F
                x >>= 5
                more = (x != -1) if (c & 0x10) else (x != 0)
                if more:
                    c |= 0x20
                out.append(chr(c + 48))
        return "".join(out)
    runs = [7, 3, 3, 3, 14]
    assert (I.rle_decode(enc(runs), 6, 5) == mask).all() and (I.rle_decode(runs, 6, 5) == mask).all()


# ============================================================================================================ flatten
def _nested(root: Path):
    _img(root / "a" / "one.jpg", seed=1)
    _lm(root / "a" / "one.json", "one.jpg")                                  # plain pair
    _img(root / "a" / "deep" / "two.jpg", seed=2)
    _lm(root / "a" / "labels" / "two.json", "../deep/two.jpg")               # imagePath relative to the JSON
    _img(root / "b" / "one.jpg", seed=3)                                      # same name, different image
    _lm(root / "b" / "one.json", "C:\\old\\one.jpg")
    raw = _img(root / "tmp.png", seed=4).read_bytes()
    (root / "tmp.png").unlink()
    _lm(root / "b" / "restored.json", "gone.png", data=base64.b64encode(raw).decode())   # image only in imageData
    _lm(root / "b" / "nothing.json", "nowhere.jpg")                           # no image, no data
    (root / "b" / "notlm.json").write_text('{"x": 1}')
    (root / "b" / "broken.json").write_text("{nope")
    _img(root / "b" / "lonely.jpg", seed=5)                                   # image without JSON
    (root / "b" / "dup_of_one.jpg").write_bytes((root / "a" / "one.jpg").read_bytes())
    _lm(root / "b" / "dup_of_one.json", "dup_of_one.jpg",
        shapes=[{"label": "bus", "shape_type": "rectangle", "points": [[0, 0], [5, 5]]}])
    (root / "b" / "notes.txt").write_text("x")
    return root


@pytest.mark.parametrize("copy", [True, False])
def test_combine_and_flatten(tmp_path, copy):
    src = _nested(tmp_path / "src")
    before = _files(src)
    o = flatten.Options([str(src / "a"), str(src / "b")], str(tmp_path / "flat"), copy=copy)
    P = flatten.plan(o)
    s = P.stats
    assert s["pairs"] == 3 and s["restore"] == 1 and s["orphan_img"] == 1 and s["dups"] == 1, dict(s)
    assert s["orphan_json"] == 3 and s["renamed"] >= 1 and s["ip_fixed"] >= 2
    assert set(P.reasons) == {flatten.R_NO_DATA, flatten.R_NOT_LABELME, flatten.R_INVALID}
    assert flatten.D_DIFF in P.dup_kinds and P.summary_rows()[0][1] == 3
    st = flatten.run(P)
    assert not st["errors"], st["errors"]
    flat = tmp_path / "flat"
    jsons = sorted(flat.glob("*.json"))
    assert len(jsons) == 4 and len(list(flat.glob("*.jpg")) + list(flat.glob("*.png"))) == 4
    for j in jsons:
        d = json.loads(j.read_text())
        assert (flat / d["imagePath"]).is_file() and d["imageData"] is None
    orph = tmp_path / "flat_orphans"
    assert len(list((orph / "jsons").glob("*.json"))) == 3 and len(list((orph / "images").iterdir())) == 1
    assert len(list((orph / "duplicates").rglob("*"))) == 2                 # the copy goes WITH its own JSON
    assert (_files(src) == before) == copy
    C.undo_run(st["undo"])
    assert _files(src) == before and _files(flat) == []


# ============================================================================================================ small tools
def test_unzip_nested_and_undo(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("deep/in.txt", "inner")
    with zipfile.ZipFile(root / "outer.zip", "w") as z:
        z.writestr("a.txt", "a")
        z.writestr("inner.zip", inner.getvalue())
    P = T.unzip_plan(root)
    assert dict(P.rows)["To extract"] == 1
    st = T.unzip_run(P, move_archives=True)
    assert st["extracted"] >= 2 and st["nested"] >= 1, st
    assert (root / "outer" / "a.txt").read_text() == "a" and not (tmp_path / "evil.txt").exists()
    assert (root / "outer" / "inner" / "deep" / "in.txt").read_text() == "inner"
    assert dict(T.unzip_plan(root).rows)["To extract"] == 0
    C.undo_run(st["undo"])
    assert (root / "outer.zip").is_file() and not (root / "outer").exists()
    with zipfile.ZipFile(root / "evil.zip", "w") as z:                          # never writes outside its folder
        z.writestr("../evil.txt", "nope")
    st = T.unzip_run(T.unzip_plan(root))
    assert "unsafe path" in " ".join(st["errors"]) and not (tmp_path / "evil.txt").exists()


def test_orphans_integrity_and_dupes(tmp_path):
    root = tmp_path / "ds"
    _img(root / "ok.jpg", seed=1)
    _lm(root / "ok.json", "ok.jpg")
    _img(root / "lonely.jpg", seed=2)
    _lm(root / "nojpg.json", "nojpg.jpg")
    _img(root / "sub" / "named.jpg", seed=3)
    _lm(root / "sub" / "other.json", "named.jpg")                              # paired by imagePath
    P = T.orphans_plan(root)
    srcs = sorted(Path(a[0]).name for a in P.actions)
    assert srcs == ["lonely.jpg", "nojpg.json"], srcs
    st = T.move_run(P, root)
    assert st["done"] == 2 and not (root / "lonely.jpg").exists()
    C.undo_run(st["undo"])
    assert (root / "lonely.jpg").exists()

    (root / "trunc.jpg").write_bytes((root / "ok.jpg").read_bytes()[:40])
    _lm(root / "trunc.json", "trunc.jpg")
    (root / "bad.json").write_text("{nope")
    _lm(root / "big.json", "ok.jpg", w=999, h=999)
    P = T.integrity_plan(root)
    bad = "\n".join(P.details["Bad files"])
    assert "trunc.jpg" in bad and "bad.json" in bad, bad
    assert any(Path(a[0]).name == "trunc.json" for a in P.actions)          # the partner goes along
    st = T.move_run(P, root)
    assert not (root / "trunc.jpg").exists() and (tmp_path / "ds_quarantine").is_dir()
    C.undo_run(st["undo"])
    assert (root / "trunc.jpg").exists()
    for p in ("trunc.jpg", "trunc.json", "bad.json", "big.json"):
        (root / p).unlink()

    for n in ("ok_1.jpg", "ok_1_2.jpg"):
        (root / n).write_bytes((root / "ok.jpg").read_bytes())
    _lm(root / "ok_1.json", "ok_1.jpg", shapes=[])
    P = T.dupes_plan(root)
    moved = sorted(Path(a[0]).name for a in P.actions)
    assert moved == ["ok_1.jpg", "ok_1.json", "ok_1_2.jpg"], moved               # kept: the annotated, shortest name
    assert P.details["Duplicates with DIFFERENT annotations — review"]
    st = T.move_run(P, root, copy=True)
    assert st["done"] == 3 and (root / "ok_1.jpg").exists()


@pytest.mark.parametrize("mode,fmt", [("long", "keep"), ("height", "jpg"), ("factor", "png")])
def test_resize_scales_labels_too(tmp_path, mode, fmt):
    src = tmp_path / "src"
    _img(src / "big.jpg", 400, 200, seed=1)
    _lm(src / "big.json", "big.jpg", w=400, h=200,
        shapes=[{"label": "car", "shape_type": "rectangle", "points": [[40, 20], [200, 100]]}])
    _img(src / "small.jpg", 50, 40, seed=2)
    o = T.ResizeOptions(str(src), str(tmp_path / "out"), mode=mode, limit=200, factor=0.5, fmt=fmt)
    with pytest.raises(ValueError):
        T.resize_plan(T.ResizeOptions(str(src), str(src / "in")))
    P = T.resize_plan(o)
    st = T.resize_run(P)
    assert not st["errors"], st
    out = tmp_path / "out"
    big = next(out.glob("big.*g"))
    h, w = cv2.imread(str(big)).shape[:2]
    assert (w, h) == (200, 100)
    d = json.loads((out / "big.json").read_text())
    assert d["imageWidth"] == 200 and d["shapes"][0]["points"] == [[20, 10], [100, 50]]
    assert d["imagePath"] == big.name
    C.undo_run(st["undo"])
    assert _files(out) == []


# ============================================================================================================ EXIF
def test_exif_rotation_baked_with_labels(tmp_path):
    from PIL import Image
    root = tmp_path / "ds"
    root.mkdir()
    a = np.zeros((40, 80, 3), np.uint8)
    a[:10, :20] = 255                                               # a mark top-left of the raw pixels
    im = Image.fromarray(a)
    exif = im.getexif()
    exif[0x0112] = 6                                                # rotate 90° clockwise to view
    im.save(root / "r.jpg", exif=exif.tobytes())
    _lm(root / "r.json", "r.jpg", w=80, h=40,
        shapes=[{"label": "m", "shape_type": "rectangle", "points": [[0, 0], [20, 10]]}])
    (root / "images").mkdir()
    (root / "labels").mkdir()
    im.save(root / "images" / "q.jpg", exif=exif.tobytes())
    (root / "labels" / "q.txt").write_text("0 0.125 0.125 0.25 0.25\n")
    im.save(root / "upright.jpg")
    _lm(root / "odd.json", "odd.jpg", w=11, h=13)
    im.save(root / "odd.jpg", exif=exif.tobytes())
    assert exif_fix.map_point(1, 3, 4, 80, 40) == (3, 4) and exif_fix.upright_size(6, 80, 40) == (40, 80)
    assert exif_fix.labels_frame(6, (80, 40), (80, 40)) == "raw" and exif_fix.labels_frame(6, (80, 40), (40, 80)) == "upright"
    P = exif_fix.plan_folder(root)
    assert sorted(Path(a[1]).name for a in P.actions) == ["q.jpg", "r.jpg"]
    assert P.details["Label size matches neither orientation — skipped (check with Image Quality)"]
    st = exif_fix.run(P, root / C.LOG_DIR)
    # LabelMe JSONs carry the size they were drawn on (raw here: rotated along); a YOLO file does not, so with
    # "auto" its labels are taken as already upright and left as they are
    assert st["done"] == 2 and st["labels"] == 1 and not st["errors"]
    with Image.open(root / "r.jpg") as r:
        assert r.size == (40, 80) and r.getexif().get(0x0112, 1) == 1
    d = json.loads((root / "r.json").read_text())
    assert d["imageWidth"] == 40 and d["imageHeight"] == 80
    xs = sorted(p[0] for p in d["shapes"][0]["points"])
    assert xs[1] == pytest.approx(40)                              # the mark is now at the top-RIGHT
    assert (root / "labels" / "q.txt").read_text() == "0 0.125 0.125 0.25 0.25\n"
    C.undo_run(st["undo"])
    with Image.open(root / "r.jpg") as r:
        assert r.size == (80, 40) and r.getexif().get(0x0112) == 6


# ============================================================================================================ pick
def test_pick_varied_images(tmp_path):
    src = tmp_path / "pool"
    for i in range(6):
        _img(src / f"red{i}.jpg", colour=(0, 0, 200 + i), seed=0)                 # near-identical
    for i, c in enumerate([(200, 0, 0), (0, 200, 0), (90, 90, 90)]):
        _img(src / f"other{i}.jpg", colour=c, seed=0)
    _lm(src / "red0.json", "red0.jpg")
    (src / "images").mkdir()
    _img(src / "images" / "y.jpg", seed=3)
    (src / "labels").mkdir()
    (src / "labels" / "y.txt").write_text("")
    paths, skipped = pick.pool(src)
    assert skipped == 2 and len(paths) == 8 and len(pick.pool(src, skip_labelled=False)[0]) == 10
    X, ok = pick.feature_matrix(paths)
    assert X.shape[0] == 8 and ok.all()
    chosen, novelty, _d = pick.k_center(X, 4, ok=ok)
    assert len(chosen) == 4 and novelty[0] != novelty[0]                       # the first pick has no novelty
    picked = [Path(paths[i]).name for i in chosen]
    assert sum(n.startswith("red") for n in picked) <= 2 and {"other0.jpg", "other1.jpg"} <= set(picked), picked
    with pytest.raises(ValueError):
        pick.plan(src, src / "out", [], [])
    P = pick.plan(src, tmp_path / "picked", [paths[0], paths[1]], [(0.5, None), (float("nan"), 0.2)])
    st = pick.run(P)
    assert st["done"] == 2 and (tmp_path / "picked" / "picked.csv").read_text().count("\n") == 3
    C.undo_run(st["undo"])
    assert _files(tmp_path / "picked") == []
