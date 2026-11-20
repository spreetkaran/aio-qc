# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Add data to an existing dataset, matched by image content: new images into LabelMe / YOLO (split) datasets with
conversion, splits and renaming; classes brought over to the images a dataset already has (LabelMe master → YOLO
subset and the other way); undo; the command line."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from AIO_QC.core.prep import add_data as A
from AIO_QC.core.prep import common as C


def _img(path: Path, seed: int, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = (np.random.default_rng(seed).random((size[1], size[0], 3)) * 255).astype("uint8")
    Image.fromarray(a).save(path)
    return path


def _lm(path: Path, shapes, w=64, h=48):
    """shapes: [(label, kind, points)]"""
    data = {"version": "5.5.0", "flags": {}, "imagePath": path.with_suffix(".png").name, "imageData": None,
            "imageWidth": w, "imageHeight": h,
            "shapes": [{"label": lab, "shape_type": kind, "points": pts, "group_id": None, "flags": {}}
                       for lab, kind, pts in shapes]}
    path.write_text(json.dumps(data), encoding="utf-8")


BOX_A = ("car", "rectangle", [[4, 4], [20, 16]])
BOX_B = ("plate", "rectangle", [[8, 10], [16, 14]])
POLY_C = ("person", "polygon", [[30, 5], [50, 5], [50, 40], [30, 40]])


def master(root: Path, n=12):
    """A LabelMe dataset with every class: img<i> has car + plate (+ person on even i)."""
    for i in range(n):
        p = _img(root / f"cam{i % 2}" / f"img{i:03d}.png", i)
        _lm(p.with_suffix(".json"), [BOX_A, BOX_B] + ([POLY_C] if i % 2 == 0 else []))
    return root


def yolo_subset(root: Path, src: Path, n=10):
    """A YOLO split dataset made from the master with the car class only, files renamed (frame_<i>)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("car\n")
    (root / "data.yaml").write_text("# my dataset\npath: .\ntrain: train/images\nval: val/images\n\nnc: 1\n"
                                    "names:\n  0: car\n")
    for i in range(n):
        sp = "val" if i % 5 == 0 else "train"
        img = root / sp / "images" / f"frame_{i}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f"cam{i % 2}" / f"img{i:03d}.png", img)
        lab = root / sp / "labels" / f"frame_{i}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("0 0.187500 0.208333 0.250000 0.250000\n")
    return root


def _plan(**kw):
    o = A.Options(**kw)
    return A.plan(o)


# ====================================================================================================== scanning
def test_scan_formats_and_layouts(tmp_path):
    m = master(tmp_path / "lm")
    (m / "cam0" / "orphan.json").write_text("{}")
    D = A.scan(m)
    assert D.fmt == "labelme" and len(D.items) == 12 and all(it.ann for it in D.items) and len(D.orphans) == 1
    y = yolo_subset(tmp_path / "y", m)
    D = A.scan(y)
    assert D.fmt == "yolo" and D.classes == ["car"] and D.layout == "split_first" and D.splits == ["train", "val"]
    assert D.image_dirs["train"] == (y / "train" / "images").resolve() and D.task == "det"
    assert {p.name for p in D.class_files} == {"data.yaml", "classes.txt"}
    # images/train layout, segmentation
    z = tmp_path / "z"
    _img(z / "images" / "valid" / "a.png", 1)
    (z / "labels" / "valid").mkdir(parents=True)
    (z / "labels" / "valid" / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
    D = A.scan(z)
    assert D.layout == "images_first" and D.split_dirs == {"val": "valid"} and D.task == "seg"
    assert A._yolo_img_dir(D, "train") == D.root / "images" / "train"


# ====================================================================================================== add new data
def test_same_annotation_within_a_pixel_and_the_dataset_line(tmp_path):
    """A LabelMe copy of a YOLO label (coordinates rounded on the way) is the same annotation, also when the
    rounding lands either side of a grid line; a real change still differs. The dataset line sees images/train."""
    a = [(0, 0.4996, 0.5, 0.2, 0.2), (1, 0.1, 0.1, 0.05, 0.05)]
    assert A._same_rows(a, [(1, 0.1, 0.1, 0.05, 0.0504), (0, 0.5004, 0.5, 0.2, 0.2)], 1 / 1000, 1)
    assert not A._same_rows(a, [(1, 0.1, 0.1, 0.05, 0.05), (0, 0.51, 0.5, 0.2, 0.2)], 1 / 1000, 1)
    assert not A._same_rows(a, a[:1], 1 / 1000, 1)
    assert A._same_rows([("car", "rectangle", 4.0, 4.0)], [("car", "rectangle", 4.4, 3.6)], 1.0, 2)
    assert not A._same_rows([("car", "rectangle", 4.0, 4.0)], [("bus", "rectangle", 4.0, 4.0)], 1.0, 2)
    from AIO_QC.widgets.data_prep import _dataset_kind
    z = tmp_path / "z"
    for sp in ("train", "val"):
        (z / "images" / sp).mkdir(parents=True)
    (z / "data.yaml").write_text("names: [a]\n")
    assert _dataset_kind(str(z)) == "YOLO · splits: train, val · class list: data.yaml"


def test_add_labelme_pairs_to_a_yolo_split_dataset(tmp_path):
    m = master(tmp_path / "master")
    y = yolo_subset(tmp_path / "yolo", m, n=10)
    new = tmp_path / "new"
    (new / "batch").mkdir(parents=True)
    for i in range(8, 12):                       # 8, 9 are already in the dataset (renamed); 10, 11 are new
        shutil.copy2(m / f"cam{i % 2}" / f"img{i:03d}.png", new / "batch" / f"x{i}.png")
        shutil.copy2(m / f"cam{i % 2}" / f"img{i:03d}.json", new / "batch" / f"x{i}.json")
    # a different image whose name clashes with a dataset file, its copy (counted once), odd files
    _lm(_img(new / "batch" / "frame_1.png", 777).with_suffix(".json"), [BOX_A, ("truck", "rectangle",
                                                                                [[1, 1], [9, 9]])])
    shutil.copy2(new / "batch" / "frame_1.png", new / "dup.png")
    shutil.copy2(new / "batch" / "frame_1.json", new / "dup.json")
    _img(new / "lonely.png", 900)                                   # no annotation
    (new / "bad.json").write_text("{nope")
    _img(new / "bad.png", 901)
    P = _plan(target=str(y), sources=[str(new)], split_mode="train", naming={"preset": "suffix"})
    assert P.target.fmt == "yolo" and P.sources[0].fmt == "labelme"
    s = P.stats
    assert s["matched"] == 2 and s["match_diff"] == 2              # the YOLO copy has only "car"
    assert s["dups_in_sources"] == 1 and s["no_annotation"] == 1 and s["invalid"] == 1
    assert len(P.adds) == 3 and s["renamed"] == 1 and sorted(P.new_classes) == ["person", "plate", "truck"]
    renamed = [a for a in P.adds if a["renamed"]][0]
    assert renamed["dst_img"].name == "frame_1_1.png" and renamed["split"] == "train"
    st = A.run(P)
    assert not st["errors"], st["errors"]
    assert st["done"]["added"] == 3 and st["done"]["classes"] == 3
    names = (y / "classes.txt").read_text().split()
    assert names[0] == "car" and set(names[1:]) == {"plate", "person", "truck"}
    import yaml
    d = yaml.safe_load((y / "data.yaml").read_text())
    assert d["nc"] == 4 and [d["names"][i] for i in range(4)] == names and d["train"] == "train/images"
    assert (y / "data.yaml").read_text().startswith("# my dataset")          # comments kept
    lab = (y / "train" / "labels" / "frame_1_1.txt").read_text().split("\n")
    ids = sorted(int(x.split()[0]) for x in lab if x)
    assert ids == [0, names.index("truck")]
    x10 = [a for a in P.adds if a["item"].path.name == "x10.png"][0]
    got = sorted(int(ln.split()[0]) for ln in x10["dst_ann"].read_text().splitlines())
    assert got == sorted([0, names.index("plate"), names.index("person")])
    assert Path(st["report"]).is_file() and "duplicate in the new data" in Path(st["report"]).read_text("utf-8-sig")
    # run it again: everything is in the dataset now
    P2 = _plan(target=str(y), sources=[str(new)])
    assert len(P2.adds) == 0 and P2.stats["matched"] == 6          # (its duplicate matches too)
    # undo: back to how it was
    C.undo_run(st["undo"])
    assert (y / "classes.txt").read_text() == "car\n" and not (y / "train" / "images" / "frame_1_1.png").exists()
    assert yaml.safe_load((y / "data.yaml").read_text())["nc"] == 1


def test_new_class_policies_balanced_split_and_hard_links(tmp_path):
    m = master(tmp_path / "master", n=30)
    y = yolo_subset(tmp_path / "yolo", m, n=5)
    new = tmp_path / "new"
    new.mkdir()
    for i in range(5, 30):
        for x in (".png", ".json"):
            shutil.copy2(m / f"cam{i % 2}" / f"img{i:03d}{x}", new / f"clip{i // 10}_frame_{i:04d}{x}")
    P = _plan(target=str(y), sources=[str(new)], new_classes="drop", split_mode="balanced", val=0.3, test=0.0)
    assert not P.new_classes and P.stats["objects_dropped"] > 0
    sp = {a["split"] for a in P.adds}
    assert sp == {"train", "val"}
    by_clip = {}
    for a in P.adds:                                        # frames of one clip stay together
        by_clip.setdefault(a["item"].path.name.split("_")[0], set()).add(a["split"])
    assert all(len(v) == 1 for v in by_clip.values())
    P = _plan(target=str(y), sources=[str(new)], new_classes="skip")
    assert len(P.adds) == 0 and P.stats["unknown_skipped"] == 25
    P = _plan(target=str(y), sources=[str(new)], new_classes="drop", link="hardlink", check_images=False)
    st = A.run(P)
    assert st["done"]["added"] == 25 and st["done"]["link"] + st["done"]["copy"] == 25
    lab = next((y / "train" / "labels").glob("clip1_*.txt")).read_text().split()
    assert lab[0] == "0" and len(lab) == 5                 # only "car" kept


def test_add_yolo_pairs_to_a_labelme_dataset_mirrored(tmp_path):
    m = master(tmp_path / "master", n=4)
    src = tmp_path / "ysrc"
    (src / "images" / "site").mkdir(parents=True)
    (src / "labels" / "site").mkdir(parents=True)
    (src / "classes.txt").write_text("plate\ncar\n")
    _img(src / "images" / "site" / "img000.png", 500)        # a new image named like one in the dataset (cam0/)
    (src / "labels" / "site" / "img000.txt").write_text("1 0.5 0.5 0.25 0.5\n0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    P = _plan(target=str(m), sources=[str(src)], subfolder="added", placement="mirror")
    assert P.target.fmt == "labelme" and len(P.adds) == 1
    a = P.adds[0]
    assert a["dst_img"] == (m / "added" / "site" / "img000.png").resolve()          # no clash in that folder
    A.run(P)
    d = json.loads(a["dst_ann"].read_text())
    assert d["imagePath"] == "img000.png" and d["imageWidth"] == 64
    kinds = sorted((s["label"], s["shape_type"]) for s in d["shapes"])
    assert kinds == [("car", "rectangle"), ("plate", "polygon")]
    car = [s for s in d["shapes"] if s["label"] == "car"][0]["points"]
    assert car == [[24.0, 12.0], [40.0, 36.0]]
    # flat placement into the dataset root: the name is taken there → renamed with the folder preset
    P = _plan(target=str(m), sources=[str(src)], placement="flat", naming={"preset": "folder"})
    assert len(P.adds) == 0                                   # already added above (the same bytes)


def test_add_with_names_clashing_in_a_labelme_folder(tmp_path):
    m = master(tmp_path / "master", n=4)
    new = tmp_path / "new" / "cam0"
    _lm(_img(new / "img000.png", 4242).with_suffix(".json"), [BOX_A])
    P = _plan(target=str(m), sources=[str(tmp_path / "new")], naming={"preset": "custom", "pattern": "{stem}_new"})
    a = P.adds[0]
    assert a["dst_img"].name == "img000_new.png" and a["dst_img"].parent.name == "cam0"
    A.run(P)
    assert json.loads(a["dst_ann"].read_text())["imagePath"] == "img000_new.png"


# ====================================================================================================== add classes
def test_bring_missing_classes_from_labelme_master_to_yolo_subset(tmp_path):
    m = master(tmp_path / "master", n=12)
    y = yolo_subset(tmp_path / "yolo", m, n=10)
    _img(y / "train" / "images" / "extra.png", 999)                # not in the master
    (y / "train" / "labels" / "extra.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    shutil.copy2(y / "train" / "images" / "frame_1.png", y / "val" / "images" / "copy_of_1.png")
    P = _plan(target=str(y), sources=[str(m)], add_new=False, on_match="merge")
    assert P.merge_keys == {"plate", "person"} and P.stats["matched"] == 10
    assert P.stats["merged_images"] == 11 and P.stats["target_unmatched"] == 1     # 10 + the copy; "extra" lacks them
    assert any("NOT have the added classes" in k for k in P.details)
    assert any("more than one split" in k for k in P.details)
    assert len(P.adds) == 0 and P.stats["not_in_target"] == 2
    st = A.run(P)
    assert not st["errors"]
    names = (y / "classes.txt").read_text().split()
    assert names == ["car", "person", "plate"] or names[0] == "car" and set(names) == {"car", "person", "plate"}
    lines = (y / "train" / "labels" / "frame_2.txt").read_text().splitlines()
    ids = sorted(int(ln.split()[0]) for ln in lines)
    assert ids == sorted([0, names.index("plate"), names.index("person")])
    assert lines[0] == "0 0.187500 0.208333 0.250000 0.250000"                    # the dataset's own line kept
    copy = (y / "val" / "labels" / "copy_of_1.txt").read_text().split()           # created for the copy
    assert int(copy[0]) == names.index("plate")
    # again: nothing left to add
    P2 = _plan(target=str(y), sources=[str(m)], add_new=False, on_match="merge")
    assert P2.merge_keys == set() and not P2.updates
    C.undo_run(st["undo"])
    assert (y / "train" / "labels" / "frame_2.txt").read_text() == "0 0.187500 0.208333 0.250000 0.250000\n"
    assert not (y / "val" / "labels" / "copy_of_1.txt").exists() and (y / "classes.txt").read_text() == "car\n"


def test_chosen_classes_conflicts_and_yolo_to_labelme(tmp_path):
    m = master(tmp_path / "master", n=4)
    y = yolo_subset(tmp_path / "yolo", m, n=4)
    # YOLO source with "car" (one box overlapping the dataset's, one new) into the LabelMe master: union adds one
    lab = y / "train" / "labels" / "frame_1.txt"
    lab.write_text("0 0.187500 0.208333 0.250000 0.250000\n0 0.8 0.8 0.1 0.1\n")
    P = _plan(target=str(m), sources=[str(y)], add_new=False, on_match="merge", merge_classes=["CAR "],
              conflict="union", iou=0.5)
    assert P.merge_keys == {"car"}
    up = {u["target"].path.name: u for u in P.updates}
    assert set(up) == {"img001.png"}                                   # only frame_1 has an extra car
    A.run(P)
    d = json.loads((m / "cam1" / "img001.json").read_text())
    cars = [s for s in d["shapes"] if s["label"] == "car"]
    assert len(cars) == 2 and d["imagePath"] == "img001.png" and len(d["shapes"]) == 3
    # replace: the dataset's cars swapped for the source's
    P = _plan(target=str(m), sources=[str(y)], add_new=False, on_match="merge", merge_classes=["car"],
              conflict="replace")
    assert len(P.updates) == 4 and P.stats["objects_replaced"] >= 4
    # keep (default): nothing to do — every image has cars
    P = _plan(target=str(m), sources=[str(y)], add_new=False, on_match="merge", merge_classes=["car"])
    assert not P.updates and P.stats["objects_kept"] > 0
    # rename map: the source's "car" is the dataset's "vehicle"
    P = _plan(target=str(m), sources=[str(y)], add_new=False, on_match="merge", rename={"car": "vehicle"})
    assert P.merge_keys == {"vehicle"} and len(P.updates) == 4


def test_replace_on_match_and_segmentation_target(tmp_path):
    m = master(tmp_path / "master", n=3)
    seg = tmp_path / "seg"
    (seg / "images").mkdir(parents=True)
    (seg / "labels").mkdir()
    (seg / "classes.txt").write_text("person\ncar\nplate\n")
    for i in range(3):
        shutil.copy2(m / f"cam{i % 2}" / f"img{i:03d}.png", seg / "images" / f"s{i}.png")
        (seg / "labels" / f"s{i}.txt").write_text("1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    P = _plan(target=str(seg), sources=[str(m)], add_new=False, on_match="replace")
    assert P.target.task == "seg" and P.stats["replaced"] == 3
    A.run(P)
    lines = (seg / "labels" / "s0.txt").read_text().splitlines()
    assert sorted(int(ln.split()[0]) for ln in lines) == [0, 1, 2]
    assert all(len(ln.split()) >= 7 for ln in lines)                    # boxes became polygons


def test_guards(tmp_path):
    m = master(tmp_path / "master", n=2)
    with pytest.raises(ValueError, match="inside the dataset"):
        _plan(target=str(m), sources=[str(m / "cam0")])
    with pytest.raises(ValueError, match="No images"):
        (tmp_path / "empty").mkdir()
        _plan(target=str(m), sources=[str(tmp_path / "empty")])
    assert A._yaml_with_names("path: .\nnames: [a, b]\nnc: 2\n", ["a", "b", "c"]).count('\n  2: "c"') == 1
    assert "nc: 3" in A._yaml_with_names("names:\n- a\n- b\nnc: 2\n", ["a", "b", "c"])


def test_cli_add_data_dry_run_and_yes(tmp_path, capsys, monkeypatch):
    from AIO_QC.core.prep import add_data_cli as CLI
    monkeypatch.chdir(tmp_path)                       # the dry run's plan report goes to the current folder
    m = master(tmp_path / "master", n=6)
    y = yolo_subset(tmp_path / "yolo", m, n=3)
    code = CLI.main(["add-data", "--dataset", str(y), "--new", str(m), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and "New images → added" in out and "dry run" in out.lower()
    assert len(list(tmp_path.glob("add_data_plan_*.csv"))) == 1
    assert len(list((y / "train" / "images").glob("*.png"))) == 2
    code = CLI.main(["add-data", "--dataset", str(y), "--new", str(m), "--yes", "--splits", "balanced",
                     "--val", "0.34", "--naming", "custom", "--pattern", "{folder}_{stem}"])
    assert code == 0
    assert len(list(y.glob("*/images/*.png"))) == 6
    code = CLI.main(["add-classes", "--dataset", str(y), "--source", str(m), "--yes"])
    out = capsys.readouterr().out
    assert code == 0 and "objects added" in out
    assert set((y / "classes.txt").read_text().split()) == {"car", "plate", "person"}
    assert CLI.main(["add-data", "--dataset", str(tmp_path / "nope"), "--new", str(m), "--yes"]) == 2


# ====================================================================================================== window
def test_data_prep_pages_add_data_and_classes(tmp_path, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from .test_aio_image import shell as _shell  # noqa: F401
    from AIO_QC.widgets.data_prep import AddClassesPage, AddDataPage, DataPrepDialog
    from AIO_QC import utils

    class _App(__import__("PyQt6.QtWidgets", fromlist=["QWidget"]).QWidget):
        settings = utils.app_settings()

        def open_dataset(self, p):
            self.opened = p
    app = _App()
    m = master(tmp_path / "master", n=12)
    y = yolo_subset(tmp_path / "yolo", m, n=8)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dp = DataPrepDialog(app)

    def go(pg, setup):
        setup(pg)
        dp._preview()
        qtbot.waitUntil(lambda: dp.job is None, timeout=60000)
        assert dp.plan is not None, dp.status.text()
        rows = {str(a).strip(): b for a, b in dp.plan.rows}
        dp._run()
        qtbot.waitUntil(lambda: dp.job is None, timeout=60000)
        return rows, dp.status.text()
    pg = dp.show_page(AddClassesPage)
    assert "YOLO" in (pg.dataset.setText(str(y)) or pg.kind.text())
    rows, msg = go(pg, lambda p: (p.sources.setPlainText(str(m)), p.classes.setText(""), p.rename.setText(""),
                                  p.add_new.setChecked(False)))
    assert rows["dataset images that gain objects"] == 8 and msg.startswith("✓"), msg
    assert set((y / "classes.txt").read_text().split()) == {"car", "plate", "person"}
    pg = dp.show_page(AddDataPage)
    rows, msg = go(pg, lambda p: (p.dataset.setText(str(y)), p.sources.setPlainText(str(m)),
                                  p.split_mode.setCurrentIndex(p.split_mode.findData("balanced")),
                                  p.like.setChecked(True), p.on_match.setCurrentIndex(0)))
    assert rows["New images → added"] == 4 and msg.startswith("✓"), msg
    assert len(list(y.glob("*/images/*.png"))) == 12
    assert not dp.undo_btn.isHidden()
    dp.close()


def test_pick_into_class_folders_keeps_same_named_images(tmp_path):
    """Pick images to label: two picks with one name in one class folder (or a name an earlier pick left there) get
    the folder prefix instead of failing."""
    from AIO_QC.core.prep import pick as PK
    pool = tmp_path / "pool"
    a = _img(pool / "camA" / "img1.png", 1)
    b = _img(pool / "camB" / "img1.png", 2)
    c = _img(pool / "camB" / "img2.png", 3)
    out = tmp_path / "picked"
    (out / "cat").mkdir(parents=True)
    _img(out / "cat" / "img2.png", 9)                         # from an earlier pick
    P = PK.plan(str(pool), str(out), [str(a), str(b), str(c)], [(0, 0)] * 3, folders=["cat", "cat", "cat"])
    dst = [Path(x[2]).name for x in P.actions]
    assert sorted(dst) == ["camA__img1.png", "camB__img1.png", "camB__img2.png"]       # every clashing one: its folder
    st = PK.run(P)
    assert st["done"] == 3 and not st["errors"]
