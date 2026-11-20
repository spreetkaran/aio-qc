# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Convert & Split (the window): side pane, formats, class order and ticks, the Save-to checks, full runs, settings."""
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _dialog(w, qtbot):
    w.open_tool("convert_yolo")
    dlg = w.open_dialogs["export"]
    assert dlg.nav.currentRow() == 0 and dlg.is_yolo()
    qtbot.waitUntil(lambda: dlg.survey is not None, timeout=20000)
    return dlg


def _set_fmt(dlg, fmt):
    dlg.fmt.setCurrentIndex(dlg.fmt.findData(fmt))
    assert dlg.fmt_key() == fmt


def test_one_window_side_pane(shell, qtbot, tmp_path):  # noqa: F811
    """Convert & Split: one window, the steps in a side pane; YOLO from any detection dataset, every format."""
    w = _open(shell, qtbot, FX.yolo(tmp_path / "y"))
    w.open_tool("split")
    dlg = w.open_dialogs["export"]
    from AIO_QC.widgets.convert_split_dialog import SECTIONS
    assert [dlg.nav.item(i).text() for i in range(dlg.nav.count())] == [s[1] for s in SECTIONS]
    assert dlg.pages.currentIndex() == dlg.nav.currentRow() == 2              # opened on its Split step
    assert dlg.is_yolo() and dlg.out.text().endswith("y_split")              # YOLO → YOLO: a split of it
    _set_fmt(dlg, "coco")
    assert dlg.out.text().endswith("y_coco")                                 # the default name follows the format
    assert dlg.merge.isHidden() and not dlg.task_hint.isHidden()           # YOLO-only options give way
    qtbot.waitUntil(lambda: "COCO" in dlg.summary.text(), timeout=5000)
    from AIO_QC.widgets.yolo_convert_dialog import ROLE_NAV_SUB
    assert dlg.nav.item(0).data(ROLE_NAV_SUB) == "COCO → y_coco"
    dlg.close()
    assert "export" not in w.open_dialogs
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_tool("export")
    dlg = w.open_dialogs["export"]
    w.open_tool("split")
    assert dlg.nav.currentRow() == 2
    for i in range(dlg.nav.count()):                                         # every step has its page
        dlg.nav.setCurrentRow(i)
        assert dlg.pages.currentIndex() == i
    titles = [t.title for t in __import__("AIO_QC.app", fromlist=["QC_TOOLS"]).QC_TOOLS]
    assert "Convert & Split" in titles and "Convert to YOLO" not in titles and "Export & Split" not in titles
    assert "Preparation" in titles and "Data Prep" not in titles
    from PyQt6.QtWidgets import QPushButton
    close = next(b for b in dlg.findChildren(QPushButton) if b.text() == "Close")
    close.click()
    assert not dlg.isVisible() and "export" not in w.open_dialogs


def test_classes_order_and_survey(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    dlg = _dialog(w, qtbot)
    names = dlg.classes()
    assert set(names) >= {"car", "plate", "wheel"}
    assert dlg.survey.suggested_task() in ("det", "seg")
    dlg.apply_order(["wheel", "bus", "car"], "a model")
    assert dlg.classes() == ["wheel", "bus", "car"]                  # "bus" is kept for its id
    from AIO_QC.widgets.yolo_convert_dialog import ROLE_NAME
    unticked = [r.data(ROLE_NAME) for r in dlg._items() if r.checkState() != Qt.CheckState.Checked]
    assert "plate" in unticked                                      # the dataset's other classes: after, unticked
    assert "not in this dataset" in dlg.status.text()
    dlg.close()


def test_save_to_checks(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    dlg = _dialog(w, qtbot)
    dlg.out.setText(str(root / "inside"))
    assert dlg.out_problem()[0] == "error"
    dlg.out.setText(str(tmp_path))                                  # contains the dataset
    assert dlg.out_problem()[0] == "error"
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x.txt").write_text("x")
    dlg.out.setText(str(busy))
    assert dlg.out_problem()[0] == "warn" and not dlg.go.isEnabled()
    dlg._use_subfolder()
    assert Path(dlg.out.text()) == busy / "ds_yolo" and dlg.out_problem()[0] == "ok"
    dlg.close()


def test_convert_runs_and_settings_are_remembered(shell, qtbot, tmp_path):  # noqa: F811
    import yaml
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    dlg = _dialog(w, qtbot)
    out = tmp_path / "yolo"
    dlg.out.setText(str(out))
    dlg.apply_order(["plate", "car"], "test")
    _set_fmt(dlg, "yolo_seg")
    dlg.start()
    qtbot.waitUntil(lambda: dlg.worker is None and dlg.open_btn.isVisible(), timeout=30000)
    y = yaml.safe_load((out / "dataset.yaml").read_text())
    assert y["names"] == {0: "plate", 1: "car"} and y["nc"] == 2
    labels = list((out / "labels").glob("*.txt"))
    assert len(labels) == 12
    for p in labels:
        for ln in p.read_text().split("\n"):
            if ln.strip():
                assert ln.split()[0] in ("0", "1") and len(ln.split()) >= 7
    assert "Converted 12 images" in dlg.status.text()
    assert dlg.out_problem()[0] == "replace"                         # a second run replaces it
    dlg.close()                                    # a page closes the whole window
    assert "export" not in w.open_dialogs
    again = _dialog(w, qtbot)
    assert again.classes() == ["plate", "car"] and again.fmt_key() == "yolo_seg" and again.out.text() == str(out)
    again.close()


def test_merge_preview_split_and_train(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    import sys
    import pytest
    if sys.platform == "win32":
        pytest.skip("fake environment is a shell script")
    from AIO_QC import utils
    from AIO_QC.core import training as T
    from .train_fake import fake_python
    root = FX.labelme(tmp_path / "ds", n=40)
    w = _open(shell, qtbot, root)
    dlg = _dialog(w, qtbot)
    car = next(it for it in dlg._items() if it.data(Qt.ItemDataRole.UserRole) == "car")
    dlg._set_out([car], "vehicle")
    assert "vehicle" in dlg.classes() and dlg.rename_map() == {"car": "vehicle"}
    dlg.out.setText(str(tmp_path / "y"))
    dlg.new.setChecked(True)
    dlg.val.setValue(20)
    dlg.g_video.setChecked(False)                          # one video otherwise: one group, nothing to split
    dlg.preview_split()
    qtbot.waitUntil(lambda: dlg.worker is None and dlg.plan is not None, timeout=30000)
    assert dlg.plan[2].report["splits"].get("val", 0) >= 4 and not dlg.preview_panel.isHidden()
    assert dlg.ftable.rowCount() > 0
    dlg.start()
    qtbot.waitUntil(lambda: dlg.worker is None and dlg.train_btn.isVisible(), timeout=30000)
    out = tmp_path / "y"
    assert (out / "val" / "labels").is_dir() and (out / "reports" / "feature_map.csv").is_file()
    import yaml
    assert "vehicle" in yaml.safe_load((out / "dataset.yaml").read_text())["names"].values()
    # train from it (fake environment)
    utils.app_settings().setValue("train/python", fake_python(tmp_path))
    dlg._train()
    td = w.train_dialog
    qtbot.waitUntil(lambda: td.check is not None and td.env is not None, timeout=60000)
    assert td.start_btn.isEnabled(), td.start_note.text()
    td.epochs.setValue(2)
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    td.start()
    run = td.current
    qtbot.waitUntil(lambda: T.state_of(T.status(run)) == "finished", timeout=60000)
    td.poll(force=True)
    assert td.b_add.isVisible() and td.t_map.value.text() != "—"
    td.tabs.setCurrentIndex(1)
    assert td.cls_table.rowCount() == 2
    td.add_to_models()
    from AIO_QC.core import model_registry as R
    assert any(m["path"].endswith("best.pt") for m in R.app_models())
    w.train_watch.tick()
    td.close()
    dlg.close()


def test_every_format_uses_the_class_list_and_can_be_undone(shell, qtbot, tmp_path):  # noqa: F811
    """Other formats: the ticked classes, renamed, in order; a new split; one set; replace and Undo."""
    import json
    root = FX.labelme(tmp_path / "ds", n=40)
    w = _open(shell, qtbot, root)
    dlg = _dialog(w, qtbot)
    dlg.apply_order(["plate", "car"], "test")
    car = next(it for it in dlg._items() if it.data(Qt.ItemDataRole.UserRole) == "car")
    dlg._set_out([car], "vehicle")
    for fmt in ("coco", "labelme", "voc", "yolo_obb", "masks", "crops", "cvat", "dota"):
        _set_fmt(dlg, fmt)
        out = tmp_path / f"o_{fmt}"
        dlg.out.setText(str(out))
        (dlg.flat if fmt != "coco" else dlg.new).setChecked(True)
        if fmt == "coco":
            dlg.val.setValue(20)
            dlg.g_video.setChecked(False)
        dlg.start()
        qtbot.waitUntil(lambda: dlg.worker is None and dlg.open_btn.isVisible(), timeout=60000)
        assert "Converted" in dlg.status.text(), (fmt, dlg.status.text())
        assert (out / "export_report.txt").is_file() and (out / ".aioqc_yolo.json").is_file()
    coco = json.loads((tmp_path / "o_coco" / "annotations" / "instances_train.json").read_text())
    assert [c["name"] for c in coco["categories"]] == ["plate", "vehicle"]
    assert (tmp_path / "o_coco" / "annotations" / "instances_val.json").is_file()
    assert (tmp_path / "o_coco" / "reports" / "split_report.txt").is_file()
    one = next((tmp_path / "o_labelme").rglob("*.json"))
    labels = {s["label"] for p in (tmp_path / "o_labelme").rglob("*.json") if not p.name.startswith(".")
              for s in json.loads(p.read_text())["shapes"]}
    assert labels <= {"plate", "vehicle"} and "car" not in labels and one
    assert {p.name for p in (tmp_path / "o_crops").iterdir() if p.is_dir()} <= {"plate", "vehicle"}
    # a second run into the folder replaces it; Undo removes exactly that
    out = tmp_path / "o_dota"
    assert dlg.out_problem()[0] == "replace"
    (out / "mine.txt").write_text("keep me")
    dlg._undo()
    assert (out / "mine.txt").is_file() and not (out / "export_report.txt").exists() and "Undone" in dlg.status.text()
    dlg.close()


def test_classification_dataset_formats(shell, qtbot, tmp_path):  # noqa: F811
    """A classifier: class folders / LabelMe / CSV, the class list (ticks, merge) applied to its classes."""
    import numpy as np
    from PIL import Image
    root = tmp_path / "cls"
    rng = np.random.default_rng(0)
    for c, n in (("cat", 12), ("dog", 12), ("bird", 3)):
        (root / c).mkdir(parents=True)
        for i in range(n):                                  # noise: no two images look alike
            Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(root / c / f"{c}_{i}.jpg")
    w = _open(shell, qtbot, root)
    w.open_tool("export")
    dlg = w.open_dialogs["export"]
    assert dlg.cls and [dlg.fmt.itemData(i) for i in range(dlg.fmt.count())] == ["cls_folders", "labelme", "cls_csv"]
    bird = next(it for it in dlg._items() if it.data(Qt.ItemDataRole.UserRole) == "bird")
    dlg._set_out([bird], "dog")                                           # merge bird into dog
    dlg.new.setChecked(True)
    dlg.val.setValue(25)
    dlg.g_video.setChecked(False)
    for fmt in ("cls_folders", "labelme", "cls_csv"):
        _set_fmt(dlg, fmt)
        dlg.out.setText(str(tmp_path / f"c_{fmt}"))
        dlg.start()
        qtbot.waitUntil(lambda: dlg.worker is None and dlg.open_btn.isVisible(), timeout=60000)
        assert "Converted 27 images" in dlg.status.text(), (fmt, dlg.status.text())
    folders = {p.name for p in (tmp_path / "c_cls_folders" / "train").iterdir()}
    assert folders == {"cat", "dog"}
    assert (tmp_path / "c_cls_folders" / "val").is_dir() and dlg.train_btn.isHidden()
    assert (tmp_path / "c_cls_csv" / "labels.csv").is_file()
    dlg.close()
