# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Image Augmentation, the window: live preview, one effect alone, recipes, Generate (with pasted objects), and the
recipe in Train Model — in memory, all the way into the training runner."""
import json
import sys

import pytest
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from AIO_QC.core import augment as A
from AIO_QC.core import augment_ops as G
from AIO_QC.core import training as T

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401
from .train_fake import fake_python, wait_ready, yolo_dataset


def _dialog(w, qtbot):
    d = w.open_augment()
    qtbot.waitUntil(lambda: not d._busy and bool(d.thumbs[1].cap.text()) and not d.workers, timeout=20000)
    return d


def _rendered(d, qtbot):
    d._schedule()
    qtbot.wait(250)
    qtbot.waitUntil(lambda: not d._busy and not d.timer.isActive(), timeout=20000)


def test_tool_is_there_and_previews(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.app import QC_TOOLS
    t = next(t for t in QC_TOOLS if t.key == "augment")
    assert t.title == "Image Augmentation" and t.shortcut == "Ctrl+Alt+A"
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    assert w.qc_actions["augment"].isEnabled() and w.a.augment.shortcut().toString() == "Ctrl+Alt+A"
    d = _dialog(w, qtbot)
    assert d is w.open_augment()                                         # one window
    assert d.thumbs[0].cap.text() == "Original" and len(d.thumbs) == 6
    assert d._src_key == w.current_key and len(d._src[1]) == 3            # this image, its three shapes
    assert "ms per image" in d.speed.text()
    assert {f.key for f in d.families} == {k for k, _t in G.FAMILIES}
    assert all("&" not in f.toggle.text().replace("&&", "") for f in d.families)
    # the recipe is what the controls say
    for k, row in d.rows.items():
        if row.on.isEnabled():
            row.on.setChecked(k == "invert")
    d.mode.setCurrentIndex(d.mode.findData("each"))
    d.rows["invert"].chance.setValue(100)
    r = d.recipe()
    assert [k for k, v in r["ops"].items() if v["on"]] == ["invert"] and r["mode"] == "each"
    _rendered(d, qtbot)
    assert all(t.cap.text() == "Invert" for t in d.thumbs[1:])
    # one effect alone, across its range
    d.rows["rotate"].eye.setChecked(True)
    _rendered(d, qtbot)
    assert d.solo_note.isVisible() and all(t.cap.text().startswith("Rotate") for t in d.thumbs[1:])
    assert d.thumbs[1].cap.text() != d.thumbs[5].cap.text()
    d.rows["rotate"].eye.setChecked(False)
    # library effects that are not installed here cannot be ticked (and say why)
    for name, info in G.libraries().items():
        fam = next(f for f in d.families if f.key == name)
        if info["version"] is None:
            assert not any(r.on.isEnabled() for r in fam.rows) and name in fam.note.text()
            assert not fam.install_btn.isHidden()                         # … with a button that installs it
        else:
            assert any(r.on.isEnabled() for r in fam.rows) and fam.install_btn.isHidden()
    # the window follows Image QC
    first = w.current_key
    d._step(1)
    assert d._src_key != first
    d.close()


def test_recipes_generate_and_paste(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    d = _dialog(w, qtbot)
    d.load_named("Light")
    assert d.recipe()["name"] == "Light" and not d.edited()
    d.rows["fog"].on.setChecked(True)
    assert d.edited()
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Yard cameras", True))
    assert d.save() == "Yard cameras" and "Yard cameras" in A.recipe_names() and not d.edited()
    assert A.load_recipe("Yard cameras")["ops"]["fog"]["on"] is True
    assert d.recipes.currentData() == "Yard cameras"
    # Generate: 2 copies, objects pasted in, originals left out
    d.tabs.setCurrentIndex(1)
    assert "12 images" in d.ds_lab.text() and d.out.text().endswith("ds_aug")
    assert not d.originals.model().item(d.originals.findData("reference")).isEnabled()    # LabelMe: no dataset.yaml
    d.copies.setValue(2)
    d.originals.setCurrentIndex(d.originals.findData("none"))
    d.paste.setChecked(True)
    d.paste_p.setValue(100)
    _rendered(d, qtbot)
    assert any(t.cap.text().startswith("Pasted") for t in d.thumbs[1:])   # the preview shows it
    d.n_workers.setValue(1)
    d._estimate()
    assert "24 new images from 12 of 12" in d.est.text() and d.go.isEnabled()
    d.out.setText(str(root / "inside"))
    d._estimate()
    assert not d.go.isEnabled() and "outside the dataset" in d.est.text()
    out = tmp_path / "ds_aug"
    d.out.setText(str(out))
    d._estimate()
    d.generate()
    qtbot.waitUntil(lambda: d.go.isVisible() and bool(d.result.text()), timeout=60000)
    assert "24 augmented images written" in d.result.text() and "objects pasted" in d.result.text()
    assert len(list(out.glob("*__aug*.json"))) == 24 and len(list(out.glob("*__aug*.jpg"))) == 24
    assert not (out / "cam_frame_0000.json").exists()
    meta = json.loads((out / ".aio_qc" / "augmentation.json").read_text())
    assert meta["recipe"]["name"] == "Yard cameras" and meta["job"]["paste"]["p"] == 1.0
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    d.generate()                                                          # again: everything is already there
    qtbot.waitUntil(lambda: d.go.isVisible() and "already there" in d.result.text(), timeout=60000)
    d.close()
    A.delete_recipe("Yard cameras")


def test_generate_yolo_from_the_window(shell, qtbot, tmp_path):  # noqa: F811
    """A YOLO dataset: only the train split, originals referred to from dataset.yaml, objects of one class pasted."""
    import yaml
    root = FX.yolo(tmp_path / "y", n=10)
    w = _open(shell, qtbot, root)
    d = _dialog(w, qtbot)
    d.load_named("Light")
    d.tabs.setCurrentIndex(1)
    assert not d.only_train.isHidden() and d.only_train.isChecked()
    i = d.originals.findData("reference")
    assert d.originals.model().item(i).isEnabled()
    d.originals.setCurrentIndex(i)
    d.copies.setValue(1)
    d.paste.setChecked(True)
    d.paste_p.setValue(100)
    d.paste_boxes.setChecked(True)
    act = next(a for a in d.paste_classes.menu().actions() if a.text() == "plate")
    act.setChecked(True)
    assert d.paste_classes.text() == "plate" and d.paste_cfg()["classes"] == ["plate"]
    d.n_workers.setValue(1)
    d._estimate()
    n_train = len(list((root / "train" / "images").glob("*.jpg")))
    assert f"{n_train} new images from {n_train} of" in d.est.text() and "referred to" in d.est.text() and d.go.isEnabled()
    d.generate()
    qtbot.waitUntil(lambda: d.go.isVisible() and bool(d.result.text()), timeout=60000)
    out = tmp_path / "y_aug"
    assert f"{n_train} augmented images written" in d.result.text() and "objects pasted" in d.result.text()
    assert len(list((out / "train" / "images").glob("*__aug1.jpg"))) == n_train and not (out / "val").exists()
    y = yaml.safe_load((out / "data.yaml").read_text())
    assert y["train"] == [str(root / "train" / "images"), str(out / "train" / "images")]
    assert y["val"] == str(root / "val" / "images")
    src = (root / "train" / "labels" / "train_0.txt").read_text().splitlines()
    new = (out / "train" / "labels" / "train_0__aug1.txt").read_text().splitlines()
    plate = str(list(y["names"].values()).index("plate"))
    assert len(new) > len(src) and {ln.split()[0] for ln in new[len(src):]} == {plate}     # only plates were pasted
    d.close()


@pytest.mark.skipif(sys.platform == "win32", reason="fake environment is a shell script")
def test_recipe_trains_in_memory(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC import utils
    utils.app_settings().setValue("train/python", fake_python(tmp_path))
    for k in ("train/hyp", "train/hyp_aug", "train/hyp_lr", "train/hyp_opt", "train/recipe", "train/extra"):
        utils.app_settings().remove(k)
    w = shell.window("image")
    yaml = yolo_dataset(tmp_path / "ds")
    w.open_training(yaml=yaml)
    td = w.train_dialog
    wait_ready(td, yaml, qtbot)
    h = td.hyp
    assert h.augment() is None and td.settings_dict()["augment"] is None and not h.recipe_share.isEnabled()
    # from Image Augmentation: "Use in Train Model"
    d = _dialog(w, qtbot)
    td._edit_recipe("Night & low light")                                  # Train Model ▸ Edit… opens it on a recipe
    assert d.recipes.currentData() == "Night & low light" and d.recipe()["name"] == "Night & low light"
    d.load_named("Outdoor camera & weather")
    assert d.recipes.currentData() == "Outdoor camera & weather"
    d.share.setValue(80)
    d.use_in_training()
    assert h.recipe_box.currentData() == "Outdoor camera & weather" and h.recipe_share.value() == 80
    aug = td.settings_dict()["augment"]
    assert aug["share"] == 0.8 and aug["recipe"]["name"] == "Outdoor camera & weather"
    assert "in memory" in h.recipe_note.text() and "recipe" in h.summary.text() and td.hyp_drawer.is_open()
    # the training environment decides which library effects exist there
    r = G.clean(dict(A.load_recipe("Light"), name="With extras"))
    r["ops"]["a_iso"]["on"] = True
    A.save_recipe(r)
    env = dict(td.env, aug_libs={"albumentations": False, "imgaug": False})
    h.set_env(env)
    h.set_recipe("With extras")
    assert "need albumentations in the training environment" in h.recipe_note.text()
    h.set_recipe("Outdoor camera & weather", 80)
    d.close()
    # a real (fake-Ultralytics) run: the runner gets the recipe and installs it
    td.epochs.setValue(1)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    td.start()
    run = td.current
    qtbot.waitUntil(lambda: T.state_of(T.status(run)) == "finished", timeout=60000)
    spec = json.loads((run / "aioqc" / "spec.json").read_text())
    assert spec["augment"]["share"] == 0.8 and spec["settings"]["augment"] == "Outdoor camera & weather"
    line = T.status(run).get("augment") or ""
    assert "on the fly" in line and "Outdoor camera & weather" in line, line
    assert (T.runner_copy().parent / "aioqc_augment.py").read_bytes() == \
        (T.RUNNER.with_name("augment_ops.py")).read_bytes()
    h.reset()
    assert h.augment() is None
    td.close()
    A.delete_recipe("With extras")


def test_generate_class_folders_boosts_rare_classes(shell, qtbot, tmp_path):  # noqa: F811
    """Class folders: rare classes get more copies (most common ÷ its count, capped), only chosen classes, and every
    copy stays in its class folder."""
    from PIL import Image
    root = tmp_path / "cf"
    for c, n in (("cat", 12), ("dog", 3), ("fox", 6)):
        (root / c).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (48, 32), (i * 20, 90, 140)).save(root / c / f"{c}{i}.png")
    w = _open(shell, qtbot, root)
    d = _dialog(w, qtbot)
    d.load_named("Light")
    d.tabs.setCurrentIndex(1)
    assert not d.balance.isHidden() and not d.classes_btn.isHidden()
    assert {a.text() for a in d.classes_btn.menu().actions()} == {"cat", "dog", "fox"}
    d.copies.setValue(1)
    d.balance.setChecked(True)
    d.cap.setValue(5)
    d._items = None
    items, copies = d._plan()
    per = {}
    for it, n in zip(items, copies):
        per.setdefault(it.img.split("/")[0], set()).add(n)
    assert per == {"cat": {1}, "dog": {4}, "fox": {2}}
    next(a for a in d.classes_btn.menu().actions() if a.text() == "dog").setChecked(True)
    items, copies = d._plan()
    assert sum(copies) == 12 and all(n == 0 for it, n in zip(items, copies) if not it.img.startswith("dog/"))
    d.n_workers.setValue(1)
    d.generate()
    qtbot.waitUntil(lambda: d.go.isVisible() and bool(d.result.text()), timeout=60000)
    out = tmp_path / "cf_aug"
    assert len(list((out / "dog").glob("*__aug*.png"))) == 12 and not list((out / "cat").glob("*__aug*"))
    d.close()
