# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Train Model ▸ Augmentation & learning rate: presets, only changed values go to Ultralytics, text-like classes never
mirror, fine-tuning, the environment's own settings, typos in extra settings."""
import json
import sys

import pytest
from PyQt6.QtWidgets import QMessageBox

from AIO_QC.core import training as T

from .train_fake import fake_python, wait_ready, yolo_dataset
from .test_aio_image import shell  # noqa: F401


def test_core_helpers():
    assert T.text_like(["car", "license_plate"]) and T.text_like(list("0123456789"))
    assert not T.text_like(["car", "truck", "person"])
    env = {"cfg_keys": ["mosaic", "fliplr"], "cfg": {"mosaic": 0.9}}
    assert T.hyp_defaults(env)["mosaic"] == 0.9 and T.hyp_defaults(None)["mosaic"] == 1.0
    assert T.unknown_settings(["mosiac", "fliplr"], env) == ["mosiac"] and T.unknown_settings(["x"], None) == []
    for p in list(T.AUG_PRESETS.values()) + list(T.LR_PRESETS.values()):
        assert set(p["values"]) <= set(T.HYP_KEYS) | {"optimizer", "lr0"}


def _dialog(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC import utils
    utils.app_settings().setValue("train/python", fake_python(tmp_path))
    for k in ("train/hyp", "train/hyp_aug", "train/hyp_lr", "train/hyp_opt", "train/hyp_lr0", "train/hyp_close_mosaic",
              "train/extra"):
        utils.app_settings().remove(k)
    w = shell.window("image")
    yaml = yolo_dataset(tmp_path / "ds")
    w.open_training(yaml=yaml)
    td = w.train_dialog                                  # (kept between tests: wait for THIS dataset's check)
    wait_ready(td, yaml, qtbot)
    return td


@pytest.mark.skipif(sys.platform == "win32", reason="fake environment is a shell script")
def test_presets_text_and_fine_tuning_reach_ultralytics(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    td = _dialog(shell, qtbot, tmp_path)
    h = td.hyp
    assert h.values() == {} and "All defaults" in h.summary.text()
    h.apply_aug("off")
    v = h.values()
    assert v["mosaic"] == 0.0 and v["fliplr"] == 0.0 and h.aug_preset.currentData() == "off"
    h.w["degrees"].setValue(7)                          # a change by hand: Custom
    assert h.aug_preset.currentData() == "custom" and h.values()["degrees"] == 7
    h.reset()
    assert h.values() == {}
    h.set_names(["car", "plate"])                       # text: never mirrored, whatever the preset
    assert h.text_safe.isChecked() and h.values() == {"fliplr": 0.0} and not h.w["fliplr"].isEnabled()
    assert h.w["fliplr"].value() == 0.0                  # shown as it will be
    h.apply_aug("strong")
    assert h.values()["fliplr"] == 0.0 and h.w["fliplr"].value() == 0.0
    h.reset()
    h.set_names(["car", "truck"])
    assert not h.text_safe.isChecked() and h.values() == {} and h.w["fliplr"].value() == 0.5
    assert not h.w["copy_paste"].isEnabled()             # segmentation only (a box dataset here)
    assert not td.lr0.isEnabled()                        # optimizer auto picks its own rate
    # fine-tune from a finished run's best.pt
    run = tmp_path / "runs" / "done"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    td.current = run
    td.fine_tune()
    s = td.settings_dict()
    assert s["optimizer"] == "AdamW" and s["lr0"] == 0.0005 and td.lr0.isEnabled()
    assert s["hyp"] == {"lrf": 0.1, "warmup_epochs": 1.0, "cos_lr": True, "freeze": 10}
    assert td.hyp_drawer.is_open() and "Fine-tuning" in td.foot_note.text()
    # a real (fake-Ultralytics) start: what the trainer is called with
    td.model_box.setCurrentIndex(max(0, td.model_box.findData("yolo11n.pt")))
    h.w["mixup"].setValue(0.2)
    td.extra.setText("fliplr=0.25")                     # typed extra settings win
    assert "win over" in td.extra_note.text()
    td.epochs.setValue(1)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    td.start()
    run2 = td.current
    qtbot.waitUntil(lambda: T.state_of(T.status(run2)) == "finished", timeout=60000)
    args = json.loads((run2 / "args.json").read_text())
    assert args["optimizer"] == "AdamW" and args["lr0"] == 0.0005 and args["freeze"] == 10 and args["cos_lr"] is True
    assert args["mixup"] == 0.2 and args["fliplr"] == 0.25 and "mosaic" not in args
    td.close()


@pytest.mark.skipif(sys.platform == "win32", reason="fake environment is a shell script")
def test_environment_decides_what_exists(shell, qtbot, tmp_path):  # noqa: F811
    td = _dialog(shell, qtbot, tmp_path)
    h = td.hyp
    env = dict(td.env)
    env["cfg_keys"] = [k for k in T.HYP_KEYS if k != "cutmix"] + ["epochs", "imgsz", "close_mosaic"]
    env["cfg"] = {"scale": 0.4}
    env["ultralytics"] = "8.1.0"
    h.w["cutmix"].setValue(0.3)
    td.env = env
    h.set_env(env)
    assert not h.w["cutmix"].isEnabled() and "cutmix" not in h.values() and "newer" in h.w["cutmix"].toolTip()
    assert h.w["scale"].value() == pytest.approx(0.4) and "scale" not in h.values()   # its own default
    td.extra.setText("mosiac=0.5")
    assert td._check_extra() is None and "mosiac" in td.extra_note.text() and not td.start_btn.isEnabled()
    td.extra.setText("")
    td.close()


@pytest.mark.skipif(sys.platform == "win32", reason="fake environment is a shell script")
def test_only_the_latest_dataset_check_counts(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    """A slow check of the dataset chosen before must not replace the result for the one chosen after it."""
    import time
    td = _dialog(shell, qtbot, tmp_path)
    a, b = yolo_dataset(tmp_path / "a"), yolo_dataset(tmp_path / "b")
    real, seen = T.check_dataset, []

    def slow_for_a(p, **kw):
        if str(p) == str(a):
            time.sleep(1.2)                               # … and finishes last
        seen.append(str(p))
        return real(p)
    monkeypatch.setattr(T, "check_dataset", slow_for_a)
    td.set_dataset(a)
    assert td.check is None and not td.start_btn.isEnabled()        # nothing to start from while it is checked
    td.set_dataset(b)
    qtbot.waitUntil(lambda: len(seen) == 2 and not td.workers, timeout=20000)
    assert seen == [str(b), str(a)] and str(td.check.yaml) == str(b) and td.yaml.text() == str(b)
    td.close()
