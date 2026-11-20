# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Combine YOLO datasets: class lists merged by name (ids rewritten only where they change), clashing names made
unique with the naming scheme, exact duplicates once, splits kept, renamed.csv, Undo — and LabelMe Combine &
Flatten with a naming scheme."""
import shutil
from pathlib import Path

from . import fixtures as FX


def _yolo(root: Path, classes, label_line, seeds=(0, 1)):
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i, seed in enumerate(seeds):
            FX._img(str(root / split / "images" / f"img{i}.jpg"), seed=seed + (50 if split == "val" else 0))
            (root / split / "labels" / f"img{i}.txt").write_text(label_line)
    (root / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    return root


def test_combine_two_yolo_datasets(tmp_path):
    from AIO_QC.core.prep import common as C, yolo_combine as YCB
    a = _yolo(tmp_path / "siteA", ["car", "plate"], "1 0.5 0.5 0.1 0.1\n")
    b = _yolo(tmp_path / "siteB", ["Plate", "truck", "car"], "0 0.5 0.5 0.1 0.1\n2 0.3 0.3 0.2 0.2\n", seeds=(0, 7))
    out = tmp_path / "combined"
    P = YCB.plan(YCB.Options([str(a), str(b)], str(out)))
    assert P.classes == ["car", "plate", "truck"]
    assert P.remap[str(b.resolve())] == {0: 1, 1: 2, 2: 0}
    assert P.stats["dups"] == 2 and P.stats["dups_diff"] == 2      # siteB/img0 = siteA/img0, but more labels
    assert P.stats["images"] == 6
    st = YCB.run(P)
    assert not st["errors"], st
    names = sorted(p.name for p in (out / "train" / "images").iterdir())
    assert names == ["img0.jpg", "siteA__img1.jpg", "siteB__img1.jpg"], names     # only clashing names change
    assert (out / "train" / "labels" / "siteB__img1.txt").read_text() == "1 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.2 0.2\n"
    assert (out / "train" / "labels" / "img0.txt").read_text() == "1 0.5 0.5 0.1 0.1\n"   # unchanged ids
    assert "names:" in (out / "dataset.yaml").read_text() and (out / "classes.txt").read_text().split() == \
        ["car", "plate", "truck"]
    assert "siteB/img1.jpg" in (out / "renamed.csv").read_text(encoding="utf-8-sig")
    assert (a / "train" / "images" / "img0.jpg").is_file()                                   # sources untouched
    u = C.undo_run(st["undo"])
    assert not u["errors"] and not any((out / "train" / "images").iterdir())


def test_flatten_labelme_with_a_naming_scheme(tmp_path):
    from AIO_QC.core.prep import flatten as F
    src = Path(FX.labelme(tmp_path / "x" / "camA", n=3))
    shutil.copytree(src, tmp_path / "x" / "camB")
    for p in (tmp_path / "x" / "camB").glob("*.jpg"):
        FX._img(str(p), seed=40 + int(p.stem[-1]))                     # different pictures, same names
    out = tmp_path / "flat"
    P = F.plan(F.Options([str(tmp_path / "x")], str(out), naming={"preset": "folder", "only_clashes": True}))
    st = F.run(P)
    assert not st["errors"], st
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert names == sorted(f"{c}__cam_frame_{i:04d}.jpg" for c in ("camA", "camB") for i in range(3)), names
    import json
    j = json.loads((out / "camB__cam_frame_0001.json").read_text())
    assert j["imagePath"] == "camB__cam_frame_0001.jpg"
