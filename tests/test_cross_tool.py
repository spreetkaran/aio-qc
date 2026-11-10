# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Phase 6: Frame Extractor -> QC, QC -> Frame Extractor, Testing -> review, one model registry, one task indicator."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from . import fixtures as FX
from .fx_fixtures import D
from .test_modes import shell3  # noqa: F401  (fixture)
from .test_testing_tab import _idle, _models_in_dataset, _no_browser, _opened  # noqa: F401


def _cli(*args):
    from AIO_QC.tools.frame_extractor.core.cli import main
    assert main(list(args)) in (0, None)


@pytest.mark.parametrize("pre", ["labelme", "yolo", "none"])
def test_open_extraction_output_in_qc(shell3, qtbot, tmp_path, pre):  # noqa: F811
    _cli("extract", os.path.join(D, "c.avi"), "-o", str(tmp_path / "out"), "-n", "15",
         "--model", os.path.join(D, "models", "dummy_a.onnx"), "--preannot", pre)
    folder = tmp_path / "out" / "c"
    n_frames = len(list(folder.rglob("*.jpg")))
    sh = shell3
    sh.show("video")
    assert sh.open_in_qc(str(folder))
    w = sh.image
    assert sh.mode == "image" and w.session is not None
    qtbot.waitUntil(lambda: w.session.index_complete, timeout=30000)
    assert len(w.session.samples) == n_frames
    labels = sum(w.session.label_totals().values())
    assert labels > 0 if pre != "none" else labels == 0
    if pre == "yolo":                                    # names from classes.txt, the .txt files kept
        assert set(w.session.label_totals()) <= {"car", "plate"}
        assert list(folder.rglob("*.txt"))


def test_show_source_video_from_qc(shell3, qtbot, tmp_path):  # noqa: F811
    sh = shell3
    sh.show("video")
    v = sh.video
    v.output_dir = str(tmp_path / "out")
    v.load_video(os.path.join(D, "a.mp4"))
    qtbot.waitUntil(lambda: v.engine is not None and v.current is not None, timeout=30000)
    for f in (25, 120):
        v.goto_frame(f)
        qtbot.waitUntil(lambda f=f: v.current.idx == f and not v.current.preview, timeout=10000)
        v.save_frame(full=True)
    v.load_video(os.path.join(D, "b.mp4"))               # another video in the player now
    qtbot.waitUntil(lambda: v.video_path.endswith("b.mp4") and v.current is not None, timeout=30000)
    assert sh.open_in_qc(str(tmp_path / "out" / "a"))
    w = sh.image
    qtbot.waitUntil(lambda: w.session.index_complete, timeout=30000)
    key = next(k for k in w.session.by_key if "_120" in k)
    w.show_source_video(key)                             # explicit user action
    assert sh.mode == "video"
    qtbot.waitUntil(lambda: v.video_path.endswith("a.mp4") and v.current_frame == 120, timeout=30000)


def test_send_failures_to_review(shell3, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import REVIEW_FLAG
    from .test_aio_image import _open
    sh = shell3
    sh.show("image")
    w = _open(sh, qtbot, FX.labelme(tmp_path / "ds"))
    _models_in_dataset(w.session.root)
    lab = w.open_testing()
    qtbot.addWidget(lab)
    _opened(lab, qtbot)
    lab.unreviewed.setChecked(True)
    lab._run_all()
    _idle(lab, qtbot)
    lab.nav.setCurrentRow(3)
    lab.tabs.setCurrentIndex([lab.tabs.tabText(i) for i in range(lab.tabs.count())].index("Failures"))
    _idle(lab, qtbot)
    assert lab._failure_items()
    lab._send_failures_to_review()
    flagged = [k for k in w.session.by_key if w.session.review.status(k) == REVIEW_FLAG]
    assert flagged
    assert w.file_dock.proxy.allowed == set(flagged) or set(flagged) <= set(w.file_dock.proxy.allowed or ())
    lab.tabs.setCurrentIndex([lab.tabs.tabText(i) for i in range(lab.tabs.count())].index("Compare A / B"))
    _idle(lab, qtbot)
    lab._send_compare_to_review()


def test_one_model_registry(shell3, qtbot, tmp_path):  # noqa: F811
    """Weights added in one place are offered everywhere; one environment setting."""
    from AIO_QC.core import model_registry as R
    from AIO_QC.tools.frame_extractor.widgets.models_dialog import load_model_cfg, save_model_cfg
    from AIO_QC.utils.qt_utils import app_settings
    from .test_aio_image import _open
    sh = shell3
    sh.show("image")
    w = _open(sh, qtbot, FX.yolo(tmp_path / "ds"))
    R.register(os.path.join(D, "models", "dummy_a.onnx"), w.session.root)        # as Models & Environment does
    assert any(m["path"].endswith("dummy_a.onnx") for m in R.app_models())
    cfg = load_model_cfg()
    cfg["models"] = [os.path.join(D, "models", "dummy_b.onnx")]                  # the Frame Extractor adds one
    cfg["env_python"] = "/opt/env/bin/python"
    save_model_cfg(cfg)
    assert app_settings().value("inference/python") == "/opt/env/bin/python"
    assert load_model_cfg()["env_python"] == "/opt/env/bin/python"
    names = {Path(m["path"]).name for m in R.all_models(w.session.root)}
    assert {"dummy_a.onnx", "dummy_b.onnx"} <= names
    lab = w.open_testing()
    qtbot.addWidget(lab)
    _opened(lab, qtbot)
    assert lab.env_edit.text() == "/opt/env/bin/python"
    assert any(Path(m["path"]).name == "dummy_b.onnx" for m in lab._registry_models())
    app_settings().remove("inference/python")


def test_one_task_indicator_in_every_mode(shell3, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets import tasks
    sh = shell3
    for m in ("image", "video", "data"):
        sh.switch(m) if sh.mode else sh.show(m)
    inds = [sh.window(m)._task_indicator for m in ("image", "video", "data")]
    assert all(not i.isVisible() for i in inds)
    cancelled = []
    t = tasks.start("Batch extraction · 2 video(s)", cancel=lambda: cancelled.append(1))
    t.update(0.5, "10 saved")
    qtbot.wait(150)
    cur = sh.window(sh.mode)._task_indicator
    assert cur.isVisible() and "50%" in cur.text()
    assert all("Batch extraction" in i.toolTip() for i in inds)
    t.cancel()
    t.finish()
    qtbot.wait(50)
    assert cancelled and "Cancelled" in cur.text()
    qtbot.waitUntil(lambda: not cur.isVisible(), timeout=4000)
    # a real job reports: the Frame Extractor's batch queue
    sh.switch("video")
    v = sh.video
    v.output_dir = str(tmp_path / "o")
    v.open_batch_queue(add_current=False)
    b = v.batch_dialog
    b.add_job(os.path.join(D, "c.avi"))
    b.start()
    seen = []
    qtbot.waitUntil(lambda: (seen.append(len(tasks.center().tasks)) or not b.is_running()), timeout=120000)
    assert max(seen) >= 1 and not tasks.center().tasks
    b.close()
