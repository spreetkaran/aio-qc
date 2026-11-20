# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Training core: dataset check, the run folder, a real detached launch (fake Ultralytics), stop / pause / resume,
failure hints, run listing."""
import json
import sys
import time

import pytest

from AIO_QC.core import training as T

from .train_fake import fake_python, yolo_dataset
from .test_aio_image import shell  # noqa: F401

posix = pytest.mark.skipif(sys.platform == "win32", reason="fake environment is a shell script")


def _wait(run, states=("finished", "stopped", "paused", "failed"), limit=60):
    t = time.time()
    while time.time() - t < limit:
        st = T.status(run)
        if T.state_of(st) in states:
            return st
        time.sleep(0.2)
    raise AssertionError(f"run stuck: {T.status(run)}\n{T.log_tail(run)[-2000:]}")


def test_check_dataset_issues_and_fixes(tmp_path):
    y = yolo_dataset(tmp_path / "ds")
    chk = T.check_dataset(y)
    assert chk.ok and chk.task == "det" and chk.n("train") == 6 and chk.n("val") == 6
    assert chk.instances["train"] == {0: 6, 1: 6}
    # moved from another computer: an absolute path that is gone
    y.write_text(y.read_text().replace("path: .", "path: /content/gone"))
    chk = T.check_dataset(y)
    assert chk.ok and chk.fixed_path and any("does not exist here" in t for _l, t in chk.issues)
    T.resolved_yaml(chk, tmp_path / "r.yaml")
    import yaml
    d = yaml.safe_load((tmp_path / "r.yaml").read_text())
    assert d["path"] == str((tmp_path / "ds").resolve()) and d["names"] == {0: "a", 1: "b"}
    # val = train, bad ids, no val
    (tmp_path / "ds" / "train" / "labels" / "train0.txt").write_text("5 0.5 0.5 0.1 0.1\n")
    y.write_text("path: .\ntrain: train/images\nval: train/images\nnames: [a, b]\n")
    chk = T.check_dataset(y)
    txt = " ".join(t for _l, t in chk.issues)
    assert "same as train" in txt and "beyond the 2 names" in txt
    y.write_text("train: train/images\nnames: [a, b]\n")
    assert not T.check_dataset(y).ok


def test_suggest_and_extra(tmp_path):
    chk = T.check_dataset(yolo_dataset(tmp_path / "ds"))
    sz, why = T.suggest_imgsz(chk)
    assert sz % 32 == 0 and why
    assert T.parse_extra("mosaic=0.5, fliplr=0 cos_lr=true name=x") == {"mosaic": 0.5, "fliplr": 0, "cos_lr": True,
                                                                         "name": "x"}
    with pytest.raises(ValueError):
        T.parse_extra("mosaic")
    assert T.model_task("yolo11s-seg.pt") == "seg" and T.model_task("yolo26n.pt") == "det"
    assert "GPU ran out of memory" in T.explain("RuntimeError: CUDA out of memory. Tried")


def _start(tmp_path, py, epochs=3, **kw):
    chk = T.check_dataset(yolo_dataset(tmp_path / "ds"))
    s = {"model": "yolo11n.pt", "epochs": epochs, "imgsz": 64, "batch": 2, "device": "cpu", "workers": 0,
         "patience": 5, "project": str(tmp_path / "runs"), "name": kw.pop("name", "r")}
    s.update(kw)
    run = T.prepare_run(py, chk, s)
    T.launch(run, py)
    return run


@posix
def test_run_to_the_end_detached(tmp_path):
    py = fake_python(tmp_path)
    info = T.probe(py)
    assert info["ultralytics"] == "8.9.0-fake" and info["devices"][0]["id"] == "cpu"
    run = _start(tmp_path, py)
    st = _wait(run)
    assert st["state"] == "finished" and st["per_class"][0]["box_map50"] == 0.5
    rows = T.epochs(run)
    assert [r["epoch"] for r in rows] == [1, 2, 3, 3] and rows[-1]["final"]
    assert T.best_scores(rows)[0] == pytest.approx(0.3)
    args = json.loads((run / "aioqc" / "spec.json").read_text())["args"]
    assert args["data"].endswith("aioqc/data.yaml") and args["name"] == "r" and args["exist_ok"]
    runs = T.list_runs(tmp_path / "runs")
    assert runs[0]["state"] == "finished" and runs[0]["best_pt"]


@posix
def test_pause_resume_stop_kill_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BATCH_S", "0.4")
    py = fake_python(tmp_path)
    run = _start(tmp_path, py, epochs=4, name="p")
    t = time.time()
    while (T.status(run).get("epoch") or 0) < 2 and time.time() - t < 30:
        time.sleep(0.2)
    T.control(run, "pause")
    st = _wait(run, ("paused",))
    assert st["resumable"]
    T.prepare_resume(run, py)
    T.launch(run, py)
    st = _wait(run, ("finished",))
    assert max(r["epoch"] for r in T.epochs(run)) == 4
    run2 = _start(tmp_path, py, epochs=4, name="s")
    time.sleep(2.5)
    T.control(run2, "stop")
    st = _wait(run2, ("stopped",))
    assert "best epoch" in st["message"] and not st["resumable"]
    run3 = _start(tmp_path, py, epochs=4, name="k")
    time.sleep(2.5)
    T.control(run3, "kill")
    st = _wait(run3, ("stopped",))
    assert st["resumable"]
    run4 = _start(tmp_path, py, epochs=99, name="f")
    st = _wait(run4, ("failed",))
    assert "out of memory" in st["error"] and T.explain(st["error"])


def test_stale_heartbeat_is_interrupted():
    st = {"state": "training", "updated": time.time() - 500, "pid": 999999}
    assert T.state_of(st) == "interrupted"
    assert T.state_of({"state": "training", "updated": time.time()}) == "training"


def test_results_csv_of_a_foreign_run(tmp_path):
    d = tmp_path / "runs" / "old"
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "best.pt").write_text("x")
    (d / "results.csv").write_text("epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),lr/pg0\n"
                                   "1,1.2,0.3,0.2,0.01\n2,1.0,0.4,0.25,0.01\n")
    r = T.list_runs(tmp_path / "runs")[0]
    assert r["state"] == "finished" and r["best"] == 0.25 and r["epoch"] == 2


def test_works_without_pyyaml(tmp_path, monkeypatch):
    """PyYAML is optional for AIO QC: the dataset check and extra settings must not need it."""
    monkeypatch.setitem(sys.modules, "yaml", None)               # `import yaml` -> ImportError
    y = yolo_dataset(tmp_path / "ds")
    y.write_text('path: .\ntrain:\n  - train/images\nval: val/images\nnames:\n  0: "a: b"\n  1: b\n')
    chk = T.check_dataset(y)
    assert chk.ok and chk.names == ["a: b", "b"] and chk.n("train") == 6
    assert T.parse_extra("mosaic=0.5, cos_lr=true freeze=[0, 1] close_mosaic=none") == \
        {"mosaic": 0.5, "cos_lr": True, "freeze": [0, 1], "close_mosaic": None}


@posix
def test_missing_ultralytics_dependency_is_caught_early(tmp_path):
    py = fake_python(tmp_path, requires=["numpy>=1.2", "aioqc-no-such-pkg>=0.20", "nvidia-ml-py>=12",
                                         'extra-only ; extra == "dev"'])
    info = T.probe(py)
    assert info["missing"] == ["aioqc-no-such-pkg"] and "nvidia-ml-py" in info["missing_optional"]
    run = _start(tmp_path, py, name="m")
    st = _wait(run, ("failed",), limit=30)
    assert "aioqc-no-such-pkg" in st["error"] and not T.epochs(run)
    assert T.missing_module(st["error"]) == "aioqc-no-such-pkg" and "pip install" in T.explain(st["error"])


@posix
def test_a_failed_run_frees_its_processes(tmp_path, monkeypatch):
    child = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_CHILD", str(child))
    py = fake_python(tmp_path)
    run = _start(tmp_path, py, epochs=99, name="crash")
    st = _wait(run, ("failed",))
    t = time.time()
    while T.pid_alive(st["pid"]) and time.time() - t < 10:
        time.sleep(0.2)
    T.reap()
    assert not T.pid_alive(st["pid"]) and T.status(run).get("released")
    t = time.time()
    while T.pid_alive(int(child.read_text())) and time.time() - t < 10:
        time.sleep(0.2)
    assert not T.pid_alive(int(child.read_text()))           # the worker went with it


@posix
def test_watchdog_kills_a_run_that_lingers_after_it_ended(tmp_path):
    import subprocess
    meta = tmp_path / "run" / "aioqc"
    meta.mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "train_runner.py"],
                         start_new_session=True)
    try:
        T._write_status(meta, {"state": "failed", "pid": p.pid, "ended": time.time() - 60, "updated": time.time()})
        assert T.is_runner(p.pid)
        assert T.ensure_released(tmp_path / "run")
        p.wait(10)
        assert not T.is_runner(p.pid)
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])   # not a runner: never
        T._write_status(meta, {"state": "failed", "pid": other.pid, "ended": time.time() - 60})
        assert not T.ensure_released(tmp_path / "run") and other.poll() is None
        other.kill()
        other.wait()
    finally:
        if p.poll() is None:
            p.kill()


@posix
def test_running_training_chip_and_end(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    """The menu-bar chip shows running training in every mode; End from its list frees the run."""
    from tests import fixtures as FX
    from tests.test_aio_image import _open
    from AIO_QC.widgets.training_monitor import monitor
    monkeypatch.setenv("FAKE_BATCH_S", "0.4")
    w = _open(shell, qtbot, FX.labelme(tmp_path / "lm"))
    chip = w.mode_switch.training_chip
    assert not chip.isVisible()
    py = fake_python(tmp_path)
    run = _start(tmp_path, py, epochs=50, name="chip")
    monitor().add(run)
    qtbot.waitUntil(lambda: chip.isVisibleTo(w.mode_switch) and "Training" in chip.text(), timeout=20000)
    from PyQt6.QtWidgets import QLabel, QMessageBox, QPushButton

    def button(c, text):
        c.open_list()
        qtbot.waitUntil(lambda: any(b.text() == text and b.isVisibleTo(c.popup)
                                    for b in c.popup.findChildren(QPushButton)), timeout=5000)
        return next(b for b in c.popup.findChildren(QPushButton) if b.text() == text and b.isVisibleTo(c.popup))
    chip.open_list()
    assert "chip" in " ".join(lab.text() for lab in chip.popup.findChildren(QLabel))
    button(chip, "Show").click()                         # Show (Image QC's own chip): Train Model on that run
    qtbot.waitUntil(lambda: getattr(w, "train_dialog", None) is not None and w.train_dialog.isVisible(), timeout=10000)
    assert w.train_dialog.current == run and not chip.popup.isVisible()
    w.train_dialog.close()
    shell.switch("data")                                 # … and from another mode: back to Image QC, the run shown
    other = shell.window("data").mode_switch.training_chip
    qtbot.waitUntil(lambda: other.isVisibleTo(shell.window("data").mode_switch), timeout=10000)
    button(other, "Show").click()
    qtbot.waitUntil(lambda: shell.mode == "image" and w.train_dialog.isVisible(), timeout=10000)
    assert w.train_dialog.current == run
    w.train_dialog.close()
    pid = T.status(run)["pid"]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    button(chip, "End").click()                          # asked, No: still running
    time.sleep(0.5)
    assert T.is_runner(pid) and not chip.popup.isVisible()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    button(chip, "End").click()
    qtbot.waitUntil(lambda: not T.is_runner(pid), timeout=20000)
    monitor().tick()
    qtbot.waitUntil(lambda: not chip.isVisibleTo(w.mode_switch), timeout=10000)
    assert T.state_of(T.status(run)) == "stopped"


def test_chart_axes_use_round_values_and_fit_their_labels(qtbot):
    """Score / loss charts: ticks on round values (no 0.00382, 0.137 …), each label drawn whole."""
    from AIO_QC.widgets.train_dialog import LineChart, nice_axis, tick_text
    a, b, s = nice_axis(0.0038, 0.538)
    assert a == 0.0 and abs(b - 0.6) < 1e-9 and s == 0.2
    assert [tick_text(a + s * i, s) for i in range(4)] == ["0.0", "0.2", "0.4", "0.6"]
    assert nice_axis(0.124, 13.7)[2] == 5 and tick_text(10, 5) == "10"
    assert [tick_text(v, 0.25) for v in (0, 0.25, 1)] == ["0.00", "0.25", "1.00"]
    c = LineChart("Scores")
    qtbot.addWidget(c)
    c.resize(400, 260)
    c.set_series([("mAP50-95", [(1, 0.0038), (2, 0.31), (3, 0.538)], "#1e88e5", False)], mark_x=3)
    c.grab()                                            # paints without error
    assert c._ystep == 0.2
