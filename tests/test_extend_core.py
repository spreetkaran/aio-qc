# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Extend Model without torch: the class plan (old ids locked, names matched, rename / merge / drop), the training
copy (hard links, labels rewritten only where ids change, old val kept apart, a leaked image left out), the plan
file, the run spec, and the before / after comparison that flags a forgotten class."""
import json
import os
from pathlib import Path

from AIO_QC.core import extend as X

from .test_combine_yolo import _yolo


def test_class_plan():
    rows, names, old_of, lm = X.plan_classes(["car", "person"], ["Car", "plate", "lorry", "junk", "truck"],
                                             {"lorry": ("merge", "truck"), "junk": ("drop",),
                                              "plate": ("rename", "licence plate")},
                                             counts_new={"Car": 4, "plate": 3, "truck": 2, "lorry": 1})
    assert names == ["car", "person", "licence plate", "truck"]
    assert old_of == {0: 0, 1: 1}
    assert lm == {"Car": 0, "plate": 2, "lorry": 3, "junk": None, "truck": 3}
    by = {(r.name, r.action): r for r in rows}
    assert by[("car", "keep")].new_images == 4 and by[("junk", "drop")].new_id is None
    assert by[("licence plate", "rename")].note == "from “plate”"
    assert ("lorry", "merge") in by


def test_training_copy_and_spec(tmp_path):
    from AIO_QC.core import training as T
    old = _yolo(tmp_path / "old", ["car", "person"], "1 0.5 0.5 0.2 0.2\n")
    new = _yolo(tmp_path / "new", ["plate", "car"], "0 0.4 0.4 0.1 0.1\n1 0.6 0.6 0.2 0.2\n", seeds=(3, 4))
    # a new training image that is an old validation image: left out (it would leak the answer)
    import shutil
    shutil.copy(old / "val" / "images" / "img0.jpg", new / "train" / "images" / "leak.jpg")
    (new / "train" / "labels" / "leak.txt").write_text("1 0.5 0.5 0.3 0.3\n")
    rows, names, old_of, lm = X.plan_classes(["car", "person"], X.yaml_names(new / "data.yaml"))
    assert names == ["car", "person", "plate"]
    P = X.ExtendPlan(base=str(tmp_path / "base.pt"), base_names=["car", "person"], names=names, old_of=old_of,
                     label_map=lm, new_yaml=str(new / "data.yaml"), old_yaml=str(old / "data.yaml"))
    est = X.estimate(P)
    assert est["old_train"] == 2 and est["new_train"] == 3
    st = X.build_tree(P, tmp_path / "copy")
    assert st["leak_skipped"] == 1 and st["images"]["old_train"] == 2 and st["images"]["new_train"] == 2
    assert st["labels_linked"] == 4 and st["labels_rewritten"] == 4          # old: same ids · new: remapped
    new_labels = sorted((tmp_path / "copy" / "labels" / "train" / "new").rglob("*.txt"))
    assert new_labels[0].read_text() == "2 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.2 0.2\n"
    old_img = next((tmp_path / "copy" / "images" / "train" / "old").rglob("img0.jpg"))
    assert os.stat(old_img).st_ino == os.stat(old / "train" / "images" / "img0.jpg").st_ino     # a hard link
    d = (tmp_path / "copy" / "data.yaml").read_text()
    assert "images/val_old" in d and "images/val_new" in d and "nc: 3" in d
    assert (tmp_path / "copy" / "val_old.yaml").is_file() and (tmp_path / "copy" / "val_new.yaml").is_file()
    # the plan file round-trips; the run spec carries it
    P.save(tmp_path / "extend.json")
    assert X.ExtendPlan.load(tmp_path / "extend.json") == P
    s = {"model": P.base, "epochs": 3, "imgsz": 320, "batch": 4, "device": "cpu", "workers": 0, "patience": 5,
         "project": str(tmp_path / "runs"), "name": "ext"}
    run, st2 = T.prepare_extend_run("python", s, P)
    spec = json.loads((run / "aioqc" / "spec.json").read_text())
    assert spec["extend"]["names"] == names and spec["args"]["data"].endswith("data.yaml")
    assert Path(spec["meta"]) == run / "aioqc" and (run / "aioqc" / "tree.json").is_file()
    assert len(json.loads((run / "aioqc" / "tree.json").read_text())["new_train_labels"]) == 2
    T.prepare_continue(run)
    assert json.loads((run / "aioqc" / "spec.json").read_text())["extend"]["review"] is False


def test_before_after_flags_a_forgotten_class():
    before = {"val_old": [{"name": "car", "map": 0.60}, {"name": "person", "map": 0.50}]}
    after = {"val_old": [{"name": "car", "map": 0.59}, {"name": "person", "map": 0.40}, {"name": "plate", "map": 0}],
             "val_new": [{"name": "plate", "map": 0.45}]}
    r = X.compare(before, after, ["car", "person"])
    assert [x["name"] for x in r["forgot"]] == ["person"]
    assert [x["name"] for x in r["learned"]] == ["plate"]


def test_classifier_copy_and_spec(tmp_path):
    from AIO_QC.core import training as T
    from . import fixtures as FX
    old, new = tmp_path / "old", tmp_path / "new"
    for sp, n in (("train", 3), ("val", 1)):
        FX.classification(old / sp, n=n)
    for sp, n in (("train", 2), ("val", 1)):
        (new / sp / "bird").mkdir(parents=True)
        for i in range(n):
            FX._img(str(new / sp / "bird" / f"b{i}.jpg"), seed=70 + i)
        FX.classification(new / sp, n=1)
    assert X.yaml_names(new) == ["bird", "cat", "dog"] and X.is_cls_folder(new)
    rows, names, old_of, lm = X.plan_classes(["cat", "dog"], X.yaml_names(new))
    P = X.ExtendPlan(base=str(tmp_path / "b.pt"), base_names=["cat", "dog"], names=names, old_of=old_of,
                     label_map=lm, new_yaml=str(new), old_yaml=str(old))
    st = X.build_cls_tree(P, tmp_path / "copy")
    assert (tmp_path / "copy" / "train" / "bird").is_dir() and st["images"]["new_train"] == 2
    assert st["leak_skipped"] == 2                         # new cat_0 / dog_0 are the old val pictures
    assert len((tmp_path / "copy" / "val_old.txt").read_text().splitlines()) == 2
    s = {"model": P.base, "epochs": 1, "imgsz": 64, "batch": 4, "device": "cpu", "workers": 0, "patience": 5,
         "project": str(tmp_path / "runs"), "name": "x"}
    run, _st = T.prepare_extend_run("python", s, P)
    spec = json.loads((run / "aioqc" / "spec.json").read_text())
    e = spec["extend"]
    assert e["cls"] and e["names"] == ["bird", "cat", "dog"] and e["old_of"] == {"1": 0, "2": 1}
    assert not e["prelabel"] and spec["args"]["data"] == str(run / "data") and spec["task"] == "cls"
    chk = T.check_dataset(old)
    assert chk.task == "cls" and chk.names == ["cat", "dog"] and chk.ok and chk.n("train") == 6
    assert T.suggest_imgsz(chk)[0] == 224 and T.weights_name("YOLO11", "n", "cls") == "yolo11n-cls.pt"
