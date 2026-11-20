# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Clean export in every format (YOLO det / seg, LabelMe, COCO, class folders — also from imageData-only JSONs), and
Split Health (findings, leakage groups, suggested moves, moving files between split folders) without the window."""
import base64
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from AIO_QC.core import dataset as D
from AIO_QC.core import export as EX
from AIO_QC.core import split_health as SH

from . import fixtures as FX


def _open(root, dtype):
    a = D.open_dataset(root, dtype)
    return a, a.discover()


@pytest.mark.parametrize("fmt", ["yolo_det", "yolo_seg", "labelme", "coco"])
def test_export_labelme_every_format(tmp_path, fmt):
    root = Path(FX.labelme(tmp_path / "ds", n=6))
    (root / "orphan.json").write_text(json.dumps({"version": "5", "shapes": [], "imagePath": "gone.jpg",
                                                  "imageData": None, "imageHeight": 0, "imageWidth": 0}))
    a, samples = _open(root, D.DatasetType.LABELME)
    by = {"train": samples[:4], "val": samples[4:], "test": []}
    out = tmp_path / fmt
    seen = []
    st = EX.export(by, a, root, out, fmt, ["car", "plate"], mode="copy", progress=lambda d, t: seen.append(d))
    assert st["images"]["train"] == 4 and st["images"]["val"] >= 2 and seen[-1] == len(samples)
    assert any("orphan" in s for s in st["skipped"])
    if fmt.startswith("yolo"):
        assert (out / "dataset.yaml").is_file() and (out / "classes.txt").read_text() == "car\nplate\n"
        lab = sorted((out / "train" / "labels").glob("*.txt"))[0].read_text().splitlines()
        assert {ln.split()[0] for ln in lab} <= {"0", "1"}
        assert st["unknown_labels"]["wheel"] >= 1                     # not in the class list: left out, counted
        if fmt == "yolo_det":
            assert all(len(ln.split()) == 5 for ln in lab)
    elif fmt == "labelme":
        j = json.loads(sorted((out / "train").glob("*.json"))[0].read_text())
        assert (out / "train" / j["imagePath"]).is_file() and j["imageData"] is None
        assert st["objects"]["wheel"] >= 4
    else:
        c = json.loads((out / "annotations" / "instances_train.json").read_text())
        assert len(c["images"]) == 4 and {x["category_id"] for x in c["annotations"]} == {1, 2}
        assert all(len(x["bbox"]) == 4 and x["area"] > 0 for x in c["annotations"])
    report = {"total": 6, "splits": {"train": 4, "val": 2}, "groups": {"multi": 0, "largest": 1, "split_across": 0},
              "classes": {"car": {"train": 4, "val": 2}}, "features": [
                  {"kind": "size", "value": "small", "images": 1, "train": 1, "val": 0, "test": 0, "status": "rare"}]}
    EX.write_reports(out, report, {s.key: "train" for s in samples[:4]}, {"format": fmt}, st)
    txt = (out / "reports" / "split_report.txt").read_text()
    assert "Train  images : 4" in txt and "size:small -> rare" in txt and "SKIPPED" in txt
    assert json.loads((out / "reports" / "split_manifest.json").read_text())["settings"] == {"format": fmt}


def test_export_name_clashes_and_cancel(tmp_path):
    root = tmp_path / "ds"
    FX.labelme(root / "a", n=2)
    FX.labelme(root / "b", n=2)                                       # the same file names in two folders
    a, samples = _open(root, D.DatasetType.LABELME)
    out = tmp_path / "out"
    st = EX.export({"train": samples}, a, root, out, "yolo_det", ["car", "plate"], mode="hardlink")
    assert st["images"]["train"] == 4 and len(list((out / "train" / "images").rglob("*.jpg"))) == 4
    assert sum(st["links"].values()) == 4
    import threading
    ev = threading.Event()
    ev.set()
    assert EX.export({"train": samples}, a, root, tmp_path / "o2", "labelme", ["car"], cancel=ev).get("cancelled")


def test_export_class_folders_also_from_image_data(tmp_path):
    root = Path(FX.classification(tmp_path / "cls", n=3))
    a, samples = _open(root, D.DatasetType.CLS_FOLDERS)
    st = EX.export({"train": samples}, a, root, tmp_path / "out", "cls_folders", ["cat", "dog"], mode="copy",
                   is_classification=True)
    assert st["images"]["train"] == 6 and len(list((tmp_path / "out" / "train" / "dog").glob("*.jpg"))) == 3
    # LabelMe flags dataset whose only copy of the image is inside the JSON
    lm = tmp_path / "flags"
    lm.mkdir()
    raw = (root / "cat" / "cat_0.jpg").read_bytes()
    for i, tag in enumerate(["cat", "dog", "cat"]):
        (lm / f"x{i}.json").write_text(json.dumps({"version": "5", "flags": {tag: True, "other": False}, "shapes": [],
                                                   "imagePath": f"missing{i}.jpg",
                                                   "imageData": base64.b64encode(raw).decode(),
                                                   "imageHeight": 240, "imageWidth": 320}))
    a, samples = _open(lm, D.DatasetType.CLS_LABELME)
    st = EX.export({"val": samples}, a, lm, tmp_path / "out2", "cls_folders", ["cat", "dog"], is_classification=True)
    assert st["links"]["decoded from imageData"] == 3 and st["images"]["val"] == 3
    assert len(list((tmp_path / "out2" / "val" / "cat").iterdir())) == 2


# ============================================================================================================ split health
def _files(spec):
    """spec: {split: [(n images, Counter per image)]}"""
    out, k = [], 0
    for sp, groups in spec.items():
        for n, c in groups:
            for _ in range(n):
                out.append((f"{sp}/img{k}.jpg", sp, Counter(c)))
                k += 1
    return out


def test_split_health_findings_and_moves():
    files = _files({"train": [(60, {"car": 1}), (30, {"truck": 2}), (5, {"bus": 1}), (5, {})],
                    "valid": [(2, {"car": 1}), (6, {"truck": 1}), (3, {"boat": 1})],
                    "test": [(1, {"car": 1})]})
    rep = SH.analyze(files, min_images=5)
    assert rep["splits"] == ["train", "val", "test"] and rep["table"]["train"]["empty"] == 5
    kinds = {(f.kind, f.split, f.cls) for f in rep["findings"]}
    assert ("missing in train", "val", "boat") in kinds
    assert ("missing in split", "val", "bus") in kinds and ("too few", "val", "car") in kinds
    assert any(k[0] == "split size" for k in kinds) and any(k[0] == "distribution drift" for k in kinds)
    assert all(f.why for f in rep["findings"])
    assert any(m.src == "train" and m.dst == "val" for m in rep["moves"])
    assert any(m.src == "val" and m.dst == "train" for m in rep["moves"])
    assert SH.js_divergence(Counter(a=1), Counter(a=1)) == pytest.approx(0)
    assert SH.js_divergence(Counter(a=1), Counter(b=1)) > 0.5
    assert SH.analyze([("a.jpg", "", Counter(car=1))])["findings"][0].kind == "no splits"


def test_split_health_leakage_groups():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 64).astype(bool)
    other = ~base
    files = [("t0", "train", Counter(car=1)), ("t1", "train", Counter(car=1)), ("v0", "val", Counter(car=1)),
             ("v1", "val", Counter(car=1))]
    near = base.copy()
    near[:3] = ~near[:3]                                              # 3 bits apart
    hashes = {"t0": base, "v0": near, "t1": other, "v1": rng.integers(0, 2, 64).astype(bool)}
    rep = SH.analyze(files, {k: np.packbits(v) for k, v in hashes.items()}, min_images=1, hamming=6)
    assert rep["groups"] and sorted(rep["groups"][0]) == ["t0", "v0"]
    assert any(f.kind == "leakage" for f in rep["findings"])
    assert any(m.key in ("t0", "v0") for m in rep["moves"])


def test_moving_files_between_split_folders(tmp_path):
    root = Path(FX.yolo(tmp_path / "y", n=2))
    a, samples = _open(root, D.DatasetType.YOLO_SPLIT)
    dirs = SH.split_dirs(root, samples)
    assert dirs == {"train": "train", "val": "val"}
    s = next(x for x in samples if x.split == "train")
    moves = SH.plan_paths(root, s, s.image_path, "val", dirs)
    assert all("val" in str(d) and not d.exists() for _s, d in moves) and len(moves) == 2
    with pytest.raises(ValueError, match="no 'test' folder"):
        SH.plan_paths(root, s, s.image_path, "test", dirs)
    assert SH.swap_split(root, tmp_path / "elsewhere.jpg", "train", "val") is None
    # LabelMe JSON moved without its image: imagePath follows
    lm = tmp_path / "lm"
    (lm / "train").mkdir(parents=True)
    (lm / "img").mkdir()
    (lm / "img" / "a.jpg").write_bytes(b"x")
    j = lm / "train" / "a.json"
    j.write_text(json.dumps({"imagePath": "../img/a.jpg", "shapes": []}))
    (lm / "val").mkdir()
    j2 = lm / "val" / "sub" / "a.json"
    j2.parent.mkdir()
    j.rename(j2)
    SH.fix_image_path(j2, lm / "img" / "a.jpg")
    assert json.loads(j2.read_text())["imagePath"] == "../../img/a.jpg"
    SH.fix_image_path(lm / "nope.json", lm / "img" / "a.jpg")                  # nothing to fix: no error


def test_export_flattened_with_the_naming_scheme(tmp_path):
    root = tmp_path / "ds"
    FX.labelme(root / "a", n=2)
    FX.labelme(root / "b", n=2)
    a, samples = _open(root, D.DatasetType.LABELME)
    out = tmp_path / "flat"
    st = EX.export({"train": samples}, a, root, out, "yolo_det", ["car", "plate"], naming={"preset": "folder"})
    imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert imgs == ["a__cam_frame_0000.jpg", "a__cam_frame_0001.jpg", "b__cam_frame_0000.jpg",
                    "b__cam_frame_0001.jpg"] and st["images"]["train"] == 4
    assert sorted(p.stem for p in (out / "train" / "labels").iterdir()) == [Path(i).stem for i in imgs]
    assert "b/cam_frame_0001,train/b__cam_frame_0001" in (out / "renamed.csv").read_text(encoding="utf-8-sig")
