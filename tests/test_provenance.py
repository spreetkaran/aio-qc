# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Provenance: shape and file records in LabelMe, the YOLO edit log, folder records and their chain, records inside
images, model records — never a person, never a secret, never a hosted model's name."""
import json
import os

import numpy as np
import pytest

from AIO_QC.core import provenance as P

from . import fixtures as FX


def _shape(label="car", pts=((10, 10), (50, 40)), st="rectangle"):
    return {"label": label, "points": [list(p) for p in pts], "group_id": None, "description": "", "shape_type": st,
            "flags": {}}


def test_new_unchanged_changed_copied_and_old_shapes():
    old = {"shapes": [_shape("car"), _shape("plate", ((1, 1), (5, 5)))]}                # written before provenance
    new = {"shapes": [_shape("car"), _shape("plate", ((1, 1), (9, 9))), _shape("bus")]}
    with P.origin("manual", "On-image edit"):
        P.stamp_labelme(new, old=old)
    car, plate, bus = new["shapes"]
    assert P.KEY not in car                                       # untouched: left as it was
    assert plate[P.KEY]["created"]["by"] == "unknown" and plate[P.KEY]["changes"][0]["what"] == ["geometry"]
    assert bus[P.KEY]["created"]["by"] == "manual" and bus[P.KEY]["created"]["tool"] == "On-image edit"
    assert new[P.KEY]["modified"]["added"] == 1 and new[P.KEY]["modified"]["changed"] == 1
    # a second save: the record is followed by its id; a relabel is one more change; a pasted copy its own record
    prev = json.loads(json.dumps(new))
    new["shapes"][2] = dict(new["shapes"][2], label="truck")
    new["shapes"].append(json.loads(json.dumps(new["shapes"][2])))
    m = P.make_origin("model", "Model Assist", model="/home/someone/weights/best.pt")
    with P.origin(m):
        P.stamp_labelme(new, old=prev)
    bus, copy = new["shapes"][2], new["shapes"][3]
    assert bus[P.KEY]["changes"][-1]["what"] == ["label"] and bus[P.KEY]["changes"][-1]["model"] == "best.pt"
    assert copy[P.KEY]["copied_from"] == bus[P.KEY]["id"] and copy[P.KEY]["id"] != bus[P.KEY]["id"]
    text = json.dumps(new)
    assert "someone" not in text and "/home" not in text           # a file name, never its folder
    assert "Made by" in P.describe_shape(bus) or "Drawn by hand" in P.describe_shape(bus)
    assert "Changed 1×" in P.describe_shape(bus)


def test_detections_keep_their_score_and_history_is_capped():
    o = P.make_origin("model", "Auto-Annotate", model="yolo11n.pt")
    sh = _shape()
    sh[P.KEY] = P.shape_record(o, score=0.91234)
    d = {"shapes": [sh]}
    P.stamp_labelme(d)
    assert d["shapes"][0][P.KEY]["created"]["score"] == 0.9123 and d[P.KEY]["created"]["by"] == "manual"
    r = d["shapes"][0][P.KEY]
    for _ in range(P.MAX_CHANGES + 5):
        r = P.bump(r, ["geometry"])
    assert len(r["changes"]) == P.MAX_CHANGES and r["changes_dropped"] == 5
    assert "Made by a model · yolo11n.pt (0.91)" in P.describe_shape({P.KEY: r})


def test_secrets_and_folders_never_written():
    o = P.make_origin("model", "AI Annotate", model=P.HOSTED)
    assert o.model == P.HOSTED and P.make_origin("model", "x", model="org/name-7b").model == "org/name-7b"
    s = P.clean_settings({"api_key": "sk-123", "proxy_key": "x", "token": "t", "password": "p",
                          "root": "/home/someone/data/ds", "win": "C:\\Users\\someone\\ds", "epochs": 5,
                          "nested": {"auth_header": "y", "lr": 0.1}})
    assert s == {"root": "ds", "win": "ds", "epochs": 5, "nested": {"lr": 0.1}}
    assert P.software().startswith("AIO QC ")


def test_labelme_adapter_records_each_save(tmp_path):
    from AIO_QC.core.dataset import LabelMeAdapter
    root = FX.labelme(tmp_path / "ds", n=3)
    ad = LabelMeAdapter(root)
    s = ad.discover()[0]
    ann = ad.load(s)
    ann.shapes[0]["points"] = [[p[0] + 3, p[1]] for p in ann.shapes[0]["points"]]
    ann.shapes.append(_shape("bus"))
    with P.origin("rule", "Overlaps"):
        ad.save(s, ann)
    d = json.loads(open(s.ann_path, encoding="utf-8").read())
    recs = [x.get(P.KEY) for x in d["shapes"]]
    assert recs[0]["changes"][0]["by"] == "rule" and recs[0]["changes"][0]["tool"] == "Overlaps"
    assert recs[-1]["created"]["tool"] == "Overlaps"
    assert sum(r is None for r in recs) == len(recs) - 2               # the others: untouched, no record
    assert ad.load(s).shapes[-1][P.KEY]["id"] == recs[-1]["id"]        # read back with the file


def test_yolo_saves_are_logged(tmp_path):
    from AIO_QC.core.dataset import YoloAdapter
    root = FX.yolo(tmp_path / "y")
    ad = YoloAdapter(root, True)
    s = ad.discover()[0]
    ann = ad.load(s)
    ann.shapes[0]["label"] = ann.shapes[0]["label"]
    del ann.shapes[0]
    with P.origin("manual", "On-image edit"):
        ad.save(s, ann)
    lines = (root / P.YOLO_LOG).read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[-1])
    assert e["removed"] == 1 and e["tool"] == "On-image edit" and e["file"].endswith(".txt")
    assert not any(p.name == P.YOLO_LOG for p in ad.discover() if hasattr(p, "name"))


def test_folder_records_chain_and_fingerprint(tmp_path):
    a = tmp_path / "raw"
    (a / "sub").mkdir(parents=True)
    (a / "sub" / "x.jpg").write_bytes(b"123")
    (a / "y.json").write_text("{}")
    r1 = P.record(a, "import", "Import", o=P.make_origin("import", "Import"))
    b = tmp_path / "out"
    r2 = P.record(b, "convert", "Convert & Split", inputs=[a], settings={"key": "secret", "val": 10})
    assert r2["inputs"][0]["provenance"] == r1["id"] and r2["inputs"][0]["fingerprint"]["files"] == 2
    assert "key" not in r2["settings"] and r2["settings"]["val"] == 10
    copy = tmp_path / "copy"
    import shutil
    shutil.copytree(a, copy)
    assert P.fingerprint(copy)["listing_sha256"] == P.fingerprint(a)["listing_sha256"]   # hidden sidecar left out
    assert len(P.read_records(b)) == 1 and P.SIDECAR.startswith(".")


def test_images_carry_their_record_and_links_stay_untouched(tmp_path):
    import cv2
    img = (np.random.default_rng(0).random((30, 40, 3)) * 255).astype("uint8")
    for ext in (".jpg", ".png"):
        ok, buf = cv2.imencode(ext, img)
        p = tmp_path / f"a{ext}"
        p.write_bytes(buf.tobytes())
        assert P.embed_file(p, P.image_record(P.make_origin("conversion", "Frame Extractor"), video="v.mp4",
                                              frame=7))
        rec = P.read_image(p)
        assert rec["video"] == "v.mp4" and rec["frame"] == 7 and rec["software"].startswith("AIO QC")
        assert cv2.imread(str(p)).shape == img.shape                   # still a valid image
        assert not P.embed_file(p, {"x": 1})                           # already has a record: unchanged
    orig = tmp_path / "orig.jpg"
    orig.write_bytes(cv2.imencode(".jpg", img)[1].tobytes())
    link = tmp_path / "link.jpg"
    os.link(orig, link)
    before = orig.read_bytes()
    assert not P.embed_file(link, {"x": 1}) and orig.read_bytes() == before


def test_switched_off_writes_nothing(tmp_path):
    P.set_enabled(False)
    try:
        d = {"shapes": [_shape()]}
        P.stamp_labelme(d)
        assert P.KEY not in d and P.KEY not in d["shapes"][0]
        assert P.record(tmp_path, "x", "y") is None and not (tmp_path / P.SIDECAR).exists()
        assert P.embed_bytes(b"\xff\xd8\xff\xd9", {"a": 1}) == b"\xff\xd8\xff\xd9"
    finally:
        P.set_enabled(True)


def test_conversions_carry_records(tmp_path):
    from AIO_QC.core import export as EX
    from AIO_QC.core.dataset import LabelMeAdapter
    root = FX.labelme(tmp_path / "ds", n=4)
    ad = LabelMeAdapter(root)
    samples = ad.discover()
    ann = ad.load(samples[0])
    with P.origin("manual", "On-image edit"):
        ann.shapes.append(_shape("car"))
        ad.save(samples[0], ann)
    P.record(root, "import", "Import")
    view = EX.ClassView(ad, ["plate", "vehicle"], {"car": "vehicle"})
    out = tmp_path / "coco"
    EX.export({"train": samples}, view, root, out, "coco", ["plate", "vehicle"], "copy")
    coco = json.loads((out / "annotations" / "instances_train.json").read_text())
    recs = [a.get(P.KEY) for a in coco["annotations"] if a.get(P.KEY)]
    assert recs and any(r["changes"][-1]["what"] == ["label"] and r["changes"][-1]["by"] == "conversion"
                        for r in recs)                               # car written as vehicle: on its record
    side = P.read_records(out)
    assert side[-1]["tool"] == "Convert & Split" and side[-1]["inputs"][0]["provenance"] == P.read_records(root)[-1]["id"]
    out2 = tmp_path / "lm"
    EX.export({"": samples}, EX.ClassView(ad, ["plate", "car"]), root, out2, "labelme", ["plate", "car"], "copy")
    j = json.loads(next(p for p in out2.rglob("*.json") if not p.name.startswith(".")).read_text())
    assert j[P.KEY]["derived"]["by"] == "conversion" and j[P.KEY]["derived"]["tool"] == "Convert & Split"


@pytest.mark.parametrize("by", P.BY)
def test_every_kind_reads_back(by):
    r = P.shape_record(P.make_origin(by, "T"))
    assert P.describe_shape({P.KEY: r})


def test_timeless_records_for_repeatable_outputs():
    o = P.make_origin("augmentation", "Image Augmentation", timeless=True)
    assert "at" not in o.entry() and "at" not in P.bump(None, ["geometry"], o)["changes"][0]
    assert "at" in P.make_origin("manual").entry()
