# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Improve Model (train → check → label more → retrain, round by round), QC presets (named checks with limits, a
pass / fail report) and plain errors (what / why / fix) — on the test datasets, models answered by a stand-in."""
import json
import os
import re
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog, QInputDialog

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


# ====================================================================================================== Improve Model
def _run(root: Path, name: str, data: bytes) -> Path:
    run = root / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(data)
    (run / "aioqc").mkdir()
    (run / "aioqc" / "spec.json").write_text(json.dumps({"names": ["car", "plate"], "task": "det",
                                                         "args": {"imgsz": 320}}))
    return run


def _predictor(good: bool):
    """Boxes on the labelled cars (a good model also finds the plates)."""
    def predict(images, ids, conf=0.05, batch=8, classes=None, cancel=None):
        out = {}
        for p in images:
            i = int(re.search(r"_(\d+)\.", os.path.basename(p)).group(1))
            x = 40 + i % 50
            boxes = [[x + 1, 41, x + 99, 119, 0.9, 0]]
            if good:
                boxes.append([x + 10, 85, x + 62, 105, 0.8, 1])
            elif i % 3 == 0:
                boxes.append([250, 10, 300, 60, 0.8, 0])          # a false alarm
            out[p] = {"w": 320, "h": 240, "by_model": {m: list(boxes) for m in ids}, "geo": {}}
        return out
    return predict


@pytest.fixture
def loop(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    runs = tmp_path / "runs"
    r1 = _run(runs, "r1", b"one")
    w.settings.setValue("train/project", str(runs))
    svc = w.inference
    monkeypatch.setattr(svc, "python", "/fake/python")
    monkeypatch.setattr(svc, "ensure_loaded", lambda specs, *a, **k: None)
    monkeypatch.setattr(svc, "predict", _predictor(False))
    return w, runs, r1, svc


def test_improve_loop_rounds(loop, qtbot, monkeypatch):
    from AIO_QC.core import improve as IM
    from AIO_QC.core.state import REVIEW_FLAG
    w, runs, r1, svc = loop
    d = w.open_improve(run=r1)
    assert d.model_path().endswith("best.pt") and d.run_dir() == r1 and d.check_btn.isEnabled()
    assert "Not checked yet" in d.check_note.text() and d.convert_btn.isVisible()        # LabelMe: convert first
    d.check()
    qtbot.waitUntil(lambda: (IM.load(w.session.root).get("history") or []) != [], timeout=60000)
    st = IM.load(w.session.root)
    assert st["model"].endswith("best.pt") and len(st["history"]) == 1
    spots = st["last_check"]["spots"]
    assert {c["label"] for c in spots["classes"]} >= {"car", "plate"}
    plate = next(c for c in spots["classes"] if c["label"] == "plate")
    assert plate["weak"] and plate["recall"] == 0 and plate["main_error"] == "miss" and "misses" in plate["advice"]
    assert d.classes.rowCount() >= 2 and d.classes.item(0, 0).text() in ("plate", "wheel")
    assert st["hard"] and d.flag_btn.isEnabled() and "plate" in IM.summary_text(spots)
    mc = w.open_dialogs["model_check"]
    assert mc.only_ids is None                                        # back to every enabled model afterwards
    # 3 · label more: flag the hardest files
    d.n_hard.setValue(5)
    d.flag_hardest()
    flagged = [k for k in w.session.by_key if w.session.review.status(k) == REVIEW_FLAG]
    assert len(flagged) == 5 and "flagged" in d.label_note.text()
    # … or pick new ones: Data Prep's Pick page, set up for this model
    d.pick()
    from AIO_QC.widgets.data_prep import PickPage
    pg = w.open_dialogs["data_prep"].page()
    assert isinstance(pg, PickPage) and pg.model_path == d.model_path() and pg.unsure.isChecked()
    assert pg.stubs.isChecked() and "to_label_" in pg.out.text()
    # 4 · retrain: Train Model, fine-tuning from it
    d.retrain()
    td = w.train_dialog
    assert td.current == r1 and td.model_box.currentData().endswith("best.pt")
    # round 2: a better model
    r2 = _run(runs, "r2", b"two")
    monkeypatch.setattr(svc, "predict", _predictor(True))
    d.refresh_models(select=str(r2 / "weights" / "best.pt"))
    assert d.run_dir() == r2
    d.check()
    qtbot.waitUntil(lambda: len(IM.load(w.session.root).get("history") or []) == 2, timeout=60000)
    st = IM.load(w.session.root)
    assert st["history"][1]["recall"] > st["history"][0]["recall"]
    assert d.history.rowCount() == 2 and "Rounds" not in d.trend.text()
    # the history is kept: a new window shows it
    d.close()
    d2 = w.open_improve()
    assert d2.history.rowCount() == 2 and d2.model_path().endswith("best.pt")
    d2.close()


def test_improve_core_bits(tmp_path):
    from AIO_QC.core import improve as IM
    st = {}
    sp = {"overall": {"precision": 0.5, "recall": 0.4, "ap5095": 0.3}, "weak": ["a"], "few": []}
    IM.record_check(st, "r1", "/m1.pt", sp, 10)
    assert IM.trend(st) == ""
    IM.record_check(st, "r2", "/m2.pt", {**sp, "overall": {**sp["overall"], "ap5095": 0.45}}, 10)
    assert "0.30 → 0.45" in IM.trend(st) and "better" in IM.trend(st)
    IM.record_check(st, "r2", "/m2.pt", {**sp, "overall": {**sp["overall"], "ap5095": 0.2}}, 10)   # same model
    assert len(st["history"]) == 2 and "worse" in IM.trend(st)
    IM.save(tmp_path, st)
    assert IM.load(tmp_path)["history"][-1]["ap5095"] == 0.2
    # empty label files
    from PIL import Image
    (tmp_path / "lm").mkdir()
    Image.new("RGB", (40, 30)).save(tmp_path / "lm" / "a.jpg")
    (tmp_path / "y" / "images").mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(tmp_path / "y" / "images" / "b.jpg")
    made = IM.label_stubs([tmp_path / "lm" / "a.jpg"])
    j = json.loads(Path(made[0]).read_text())
    assert j["shapes"] == [] and j["imageWidth"] == 40 and j["imagePath"] == "a.jpg"
    assert IM.label_stubs([tmp_path / "lm" / "a.jpg"]) == []                   # never overwritten
    made = IM.label_stubs([tmp_path / "y" / "images" / "b.jpg"], "yolo")
    assert made == [str(tmp_path / "y" / "labels" / "b.txt")]
    assert IM.summary_text({"overall": {}, "weak": [], "few": []}).endswith("no weak class at these thresholds.")


def test_train_dialog_offers_improve(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    r = _run(tmp_path / "runs", "r1", b"x")
    w.open_training(run=r)
    td = w.train_dialog
    td.current = r
    td._show_run_buttons({}, "finished", r)
    assert not td.b_improve.isHidden()
    td.close()


# ====================================================================================================== QC presets
def test_qc_preset_runs_and_reports(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core import qc_presets as QP
    from AIO_QC.core.state import REVIEW_FLAG
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_tool("qc_presets")
    d = w.open_dialogs["qc_presets"]
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Delivery", True))
    d.new_preset()
    assert d.current()["name"] == "Delivery"
    for key in ("validation", "text_labels", "outliers"):
        d.tool_box.setCurrentIndex(d.tool_box.findData(key))
        d.add_check()
    p = d.current()
    assert [c["tool"] for c in p["checks"]] == ["validation", "text_labels", "outliers"]
    assert p["checks"][2]["params"]                                     # the window's settings were taken
    assert "outliers" not in w.open_dialogs                            # (hidden windows, closed again)
    d.checks.cellWidget(2, 1).setValue(1000)                            # outliers: allowed up to 1000 files
    assert QP.load()[0]["checks"][2]["max_files"] == 1000
    d.start()
    qtbot.waitUntil(lambda: d.run is None, timeout=120000)
    res = {r["tool"]: r for r in d.results}
    assert set(res) == {"validation", "text_labels", "outliers"} and not any(r["error"] for r in d.results), d.results
    assert res["validation"]["passed"] and res["validation"]["files"] == 0
    assert res["text_labels"]["files"] >= 1 and not res["text_labels"]["passed"]      # the odd plate texts
    assert res["outliers"]["files"] >= 1 and res["outliers"]["passed"]
    assert QP.verdict(d.results) == "FAIL" and "FAIL" in d.verdict.text()
    html, csvp = d.report
    assert html.is_file() and "What to fix" in html.read_text() and csvp.read_text().count("\n") > 1
    assert QP.load()[0]["last"]["verdict"] == "FAIL" and "FAIL" in d.listw.item(0).text()
    n = d.flag_failing()
    assert n >= 1 and sum(w.session.review.status(k) == REVIEW_FLAG for k in w.session.by_key) == n
    # the preset keeps its settings: a changed window does not change the next run
    w.open_tool("validation")
    v = w.open_dialogs["validation"]
    from AIO_QC.widgets.qc_presets_dialog import apply_settings, capture
    before = capture(v)
    assert apply_settings(v, p["checks"][0]["params"]) == 0 and capture(v) == before
    d.checks.selectRow(0)
    d.take_settings()
    d.open_check()
    v.close()
    # share: export, import (a new name)
    f = tmp_path / "share.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(f), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
    d.export_preset()
    assert "last" not in json.loads(f.read_text())["preset"]
    d.import_preset()
    assert [x["name"] for x in d.presets] == ["Delivery", "Delivery (2)"]
    d.copy_preset()
    assert d.current()["name"] == "Delivery (2) (copy)"
    d.delete_preset()
    assert len(d.presets) == 2
    d.checks.selectRow(0)
    d.remove_check()
    assert len(d.current()["checks"]) == 2
    d.close()


def test_qc_preset_core(tmp_path):
    from AIO_QC.core import qc_presets as QP
    c = QP.new_check("validation", "Validation")
    rows = [("a", "error", "x"), ("a", "warn", "y"), ("b", "warn", "z")]
    r = QP.evaluate(c, rows)
    assert r["files"] == 2 and not r["passed"]
    c.update(max_files=2)
    assert QP.evaluate(c, rows)["passed"]
    c.update(max_files=0, count="errors")
    r = QP.evaluate(c, rows)
    assert r["files"] == 1 and r["findings"] == 1
    assert not QP.evaluate(c, [], "broke")["passed"] and QP.verdict([QP.evaluate(c, [], "broke")]) == "incomplete"
    assert QP.verdict([]) == "no checks"
    bad = tmp_path / "bad.json"
    bad.write_text('{"x": 1}')
    with pytest.raises(ValueError):
        QP.import_preset(bad)
    assert QP.unique_name("A", [{"name": "a"}, {"name": "A (2)"}]) == "A (3)"
    h, cs = QP.write_report(tmp_path, {"name": "P/1", "note": "n", "checks": [c]}, [QP.evaluate(c, rows)], "/ds")
    assert h.name.startswith("P_1_") and "<li class=error>x</li>" in h.read_text()


# ====================================================================================================== plain errors
@pytest.mark.parametrize("err,what,action", [
    (ValueError("Add at least one model in ⚙ Models & environment."), "Add at least one model", "model_setup"),
    (ModuleNotFoundError("No module named 'yaml'"), "missing the package", "install:pyyaml"),
    (PermissionError("[WinError 32] The process cannot access the file because it is being used by another "
                     "process"), "open in another program", "retry"),
    (OSError("[Errno 28] No space left on device"), "The disk is full", ""),
    (FileNotFoundError("[Errno 2] No such file or directory: 'x.json'"), "was not found", ""),
    ("RuntimeError: CUDA out of memory. Tried to allocate 2 GiB", "graphics card ran out of memory", "model_setup"),
    (json.JSONDecodeError("Expecting value", "x", 0), "JSON file is damaged", ""),
    (RuntimeError("something odd"), "something odd", "open_log"),
])
def test_explain(err, what, action):
    from AIO_QC.core.explain import explain
    e = explain(err)
    assert what in e.what and e.action == action, e


def test_show_error_box_and_fix(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.widgets import errors as ER
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    opened = []
    monkeypatch.setattr(w, "open_model_setup", lambda: opened.append("setup"))
    e = ER.show_error(w, "Test", ValueError("Choose a Python environment in ⚙ Models & environment."))
    assert e.action == "model_setup" and ER.LAST[-1] is e
    assert ER.run_fix(e, w) and opened == ["setup"]
    tried = []
    e = ER.show_error(w, "Test", OSError("WinError 32 being used by another process"), retry=lambda: tried.append(1))
    assert ER.run_fix(e, w, lambda: tried.append(1)) and tried == [1]
    assert "Windows does not let" in ER.message(OSError("being used by another process"))
    # a scan that cannot start says so plainly (and offers the fix) — Model Check without models
    w.open_tool("model_check")
    mc = w.open_dialogs["model_check"]
    mc.start_scan()
    assert "model" in ER.LAST[-1].what.lower() and ER.LAST[-1].action == "model_setup"
    mc.close()
