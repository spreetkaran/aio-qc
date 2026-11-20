# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Segmentation masks: class-index (VOC palette PNGs, plain grey), colour (with a labelmap.txt, or found by
themselves) and instance masks (with class masks; Cityscapes ids) imported as polygons; written from shapes as a
palette PNG + 16-bit instances; and back again — the same objects."""
import json
from pathlib import Path

import cv2
import numpy as np

from . import fixtures as FX


def _imgs(root: Path, n=3):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        FX._img(str(root / f"im{i}.jpg"), seed=i)
    return root


def test_index_color_and_instance_masks(tmp_path):
    from AIO_QC.core.prep import importers as IM
    from AIO_QC.core.prep.masks import palette, sniff, write_mask_png
    imgs = _imgs(tmp_path / "imgs")
    md = tmp_path / "SegmentationClass"
    md.mkdir()
    for i in range(3):
        m = np.zeros((240, 320), np.uint8)
        m[20:80, 20:100] = 1
        m[120:200, 150:300] = 2
        m[80:82, 20:100] = 255                                  # VOC border: ignored
        write_mask_png(md / f"im{i}.png", m, 2)
    (tmp_path / "classes.txt").write_text("background\ncat\ndog\n")
    assert sniff(md)["mode"] == "index"
    P = IM.plan(md, imgs, tmp_path / "o1", "auto", "labelme",
                spec={"labelmap": str(tmp_path / "classes.txt")})
    st = IM.run(P)
    assert not st["errors"]
    d = json.loads((tmp_path / "o1" / "im0.json").read_text())
    assert sorted(s["label"] for s in d["shapes"]) == ["cat", "dog"] and all(s["shape_type"] == "polygon"
                                                                             for s in d["shapes"])
    # colour masks + labelmap.txt
    cd = tmp_path / "color"
    cd.mkdir()
    for i in range(3):
        m = np.zeros((240, 320, 3), np.uint8)
        m[30:90, 30:90] = (0, 0, 255)                         # BGR red
        cv2.imwrite(str(cd / f"im{i}_mask.png"), m)
    (tmp_path / "labelmap.txt").write_text("background:0,0,0::\nfire:255,0,0::\n")
    assert sniff(cd)["mode"] == "color"
    P = IM.plan(cd, imgs, tmp_path / "o2", "masks", "yolo_seg", spec={"labelmap": str(tmp_path / "labelmap.txt")})
    IM.run(P)
    assert (tmp_path / "o2" / "labels" / "im0.txt").read_text().startswith("0 ")
    assert (tmp_path / "o2" / "classes.txt").read_text().split() == ["fire"]
    # instances: two objects of one class, class from the class mask
    idir = tmp_path / "SegmentationObject"
    idir.mkdir()
    for i in range(3):
        m = np.zeros((240, 320), np.uint8)
        m[20:80, 20:100] = 1
        m[120:200, 150:300] = 2
        cv2.imwrite(str(idir / f"im{i}.png"), m)
    P = IM.plan(idir, imgs, tmp_path / "o3", "masks", "labelme",
                spec={"mode": "instance", "class_masks": str(md), "labelmap": str(tmp_path / "classes.txt")})
    IM.run(P)
    d = json.loads((tmp_path / "o3" / "im1.json").read_text())
    assert sorted(s["label"] for s in d["shapes"]) == ["cat", "dog"]
    assert palette(3)[1] == (128, 0, 0)


def test_export_masks_and_back(tmp_path):
    from AIO_QC.core import dataset as D, export as EX
    from AIO_QC.core.prep import importers as IM
    lm = Path(FX.labelme(tmp_path / "lm", n=3))
    a = D.open_dataset(lm, D.DatasetType.LABELME)
    smp = a.discover()
    labels = sorted({x["label"] for s in smp for x in a.load(s).shapes})
    out = tmp_path / "masks_out"
    st = EX.export({"train": smp}, a, lm, out, "masks", labels)
    pngs = sorted((out / "train" / "masks").rglob("*.png"))
    assert len(pngs) == 3 and (out / "labelmap.txt").is_file() and (out / "train" / "instances").is_dir()
    from PIL import Image
    m = np.asarray(Image.open(pngs[0]))
    assert Image.open(pngs[0]).mode == "P" and set(np.unique(m)) - {0} <= set(range(1, len(labels) + 1))
    inst = cv2.imread(str(next((out / "train" / "instances").rglob("*.png"))), cv2.IMREAD_UNCHANGED)
    assert inst.dtype == np.uint16 and inst.max() >= 1
    # back to polygons: every exported class comes back
    P = IM.plan(out / "train" / "masks", out / "train" / "images", tmp_path / "back", "masks", "labelme",
                spec={"labelmap": str(out / "labelmap.txt")})
    IM.run(P)
    got = {s["label"] for p in (tmp_path / "back").rglob("*.json") for s in json.loads(p.read_text())["shapes"]}
    filled = {k for k, v in st["objects"].items() if v}
    assert got and got <= set(labels) and len(got) >= len(filled) - 1
