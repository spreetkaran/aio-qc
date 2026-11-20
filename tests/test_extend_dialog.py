# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Extend Model in the window: the plan dialog (the model's classes locked, the new data's classes matched or
added, merge / leave out, completeness, the old data found from the model's own run), Train Model's Goal, the run
it prepares (launch stubbed), the review step's buttons, the before / after summary, Improve ▸ Extend."""
import json
from pathlib import Path

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401
from .test_combine_yolo import _yolo


def _base_run(tmp_path, old_yaml):
    run = tmp_path / "runs" / "base"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    (run / "aioqc").mkdir()
    (run / "aioqc" / "spec.json").write_text(json.dumps({"names": ["car", "person"], "dataset_yaml": str(old_yaml)}))
    (run / "aioqc" / "status.json").write_text(json.dumps({"state": "finished", "epoch_time": 30,
                                                           "train_images": 2}))
    return run


def test_extend_plan_dialog(qtbot, tmp_path):
    from AIO_QC.core import extend as X
    from AIO_QC.widgets.extend_dialog import ExtendDialog
    old = _yolo(tmp_path / "old", ["car", "person"], "1 0.5 0.5 0.2 0.2\n")
    new = _yolo(tmp_path / "new", ["plate", "Car", "junk"], "0 0.4 0.4 0.1 0.1\n1 0.6 0.6 0.2 0.2\n", seeds=(3, 4))
    run = _base_run(tmp_path, old / "data.yaml")
    d = ExtendDialog(None, str(run / "weights" / "best.pt"), str(new / "data.yaml"))
    qtbot.addWidget(d)
    assert d.base_names == ["car", "person"] and d.old_yaml.text() == str(old / "data.yaml")   # from its run
    P = d.plan()
    assert P.names == ["car", "person", "plate", "junk"] and P.old_of == {0: 0, 1: 1}
    assert "person" in d.complete.text() and "pre-labelled" in d.complete.text()
    assert "old + " in d.summary.text() and "per epoch" in d.summary.text()
    d._act("junk", (X.DROP,))
    d._act("plate", (X.MERGE, "car"))
    P = d.plan()
    assert P.names == ["car", "person"] and P.label_map["junk"] is None and P.label_map["plate"] == 0
    assert not d.ok.isEnabled()                       # nothing new to learn for "new classes"
    d.goal.setCurrentIndex(d.goal.findData("images"))
    assert d.ok.isEnabled()
    d.actions.clear()
    d.goal.setCurrentIndex(d.goal.findData("classes"))
    d.prelabel.setChecked(False)
    d._refresh()
    assert "NOT labelled" in d.complete.text()
    d.old_yaml.clear()
    assert "may forget" in d.summary.text()
    d._accept()
    assert d.result_plan is not None and d.result_plan.old_yaml == ""


def test_train_dialog_extends_a_model(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core import extend as X, training as T
    from AIO_QC.widgets import extend_dialog as ED
    old = _yolo(tmp_path / "old", ["car", "person"], "1 0.5 0.5 0.2 0.2\n")
    new = _yolo(tmp_path / "new", ["plate"], "0 0.4 0.4 0.1 0.1\n", seeds=(3, 4))
    run = _base_run(tmp_path, old / "data.yaml")
    best = str(run / "weights" / "best.pt")
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_training(yaml=str(new / "data.yaml"))
    td = w.train_dialog
    qtbot.waitUntil(lambda: td.check is not None, timeout=30000)
    td.env = {"_python": "python", "torch": "2", "ultralytics": "8"}
    td.model_box.insertItem(0, "base", best)
    td.model_box.setCurrentIndex(0)
    td.goal.setCurrentIndex(td.goal.findData("classes"))
    assert not td.extend_btn.isHidden()
    # the plan (the dialog is answered as a person would: accepted with its defaults)
    monkeypatch.setattr(ED.ExtendDialog, "exec", lambda self: (self._accept(), 1)[1])
    assert td.plan_extend() and td.extend_plan.new_classes == ["plate"]
    assert "1 new (plate)" in td.model_note.text()
    launched = []
    monkeypatch.setattr(T, "launch", lambda r, py: launched.append(r) or {"survives": True})
    td.project.setText(str(tmp_path / "runs"))
    td.start()
    qtbot.waitUntil(lambda: bool(launched), timeout=30000)
    r = launched[0]
    spec = json.loads((r / "aioqc" / "spec.json").read_text())
    assert spec["extend"]["names"] == ["car", "person", "plate"] and (r / "data" / "data.yaml").is_file()
    assert (r / "aioqc" / "README.txt").is_file() and (tmp_path / "runs" / "lineage.jsonl").is_file()
    # the review step: its buttons, then Continue
    td.current = r
    td._show_run_buttons({"state": "review"}, "review", r)
    assert not td.b_review.isHidden() and not td.b_continue.isHidden()
    td.continue_extend()
    assert json.loads((r / "aioqc" / "spec.json").read_text())["extend"]["review"] is False and len(launched) == 2
    # finished: the before / after summary
    (r / "aioqc" / "compare.json").write_text(json.dumps({
        "before": {"val_old": [{"name": "car", "map": 0.6}, {"name": "person", "map": 0.5}]},
        "after": {"val_old": [{"name": "car", "map": 0.58}, {"name": "person", "map": 0.3}],
                  "val_new": [{"name": "plate", "map": 0.4}]}}))
    html = td._extend_summary(r, spec)
    assert "learned plate 40" in html and "person -20" in html
    assert X.ExtendPlan.load(r / "aioqc" / "extend.json").names == ["car", "person", "plate"]
    td.goal.setCurrentIndex(td.goal.findData("train"))
    (r / "aioqc" / "status.json").write_text(json.dumps({"state": "finished"}))   # (the app-wide monitor: not "running")
    td.close()


def test_improve_offers_extend(shell, qtbot, tmp_path):  # noqa: F811
    old = _yolo(tmp_path / "old", ["car", "person"], "1 0.5 0.5 0.2 0.2\n")
    run = _base_run(tmp_path, old / "data.yaml")
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_improve()
    d = w.open_dialogs["improve"]
    d.model_path = lambda: str(run / "weights" / "best.pt")
    d.extend()
    td = w.train_dialog
    assert td.goal.currentData() == "classes" and td.model_box.currentData() == str(run / "weights" / "best.pt")
    td.close()
    d.close()
    assert Path(run).is_dir()
