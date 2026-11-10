# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Model Lab on real Qt: test set, models, run (images + video), every results tab, report, project round trip."""
from __future__ import annotations

import os
import shutil

import pytest

from .fx_fixtures import D  # noqa: E402  (generated media + models)


@pytest.fixture
def lab(qapp, qtbot, tmp_path, monkeypatch):
    from AIO_QC.widgets.testing import lab_common, model_lab
    monkeypatch.setattr(lab_common, "open_path", lambda *a, **k: None)
    monkeypatch.setattr(model_lab, "open_path", lambda *a, **k: None)
    lab = model_lab.ModelLab(None, {"models": []}, lambda: None, "test")
    qtbot.addWidget(lab)
    lab.show()
    qtbot.waitUntil(lambda: lab.job is None, timeout=60000)      # the restored session finishes loading
    yield lab
    lab.close()


def _wait_idle(lab, qtbot, t=120000):
    qtbot.waitUntil(lambda: lab.job is None, timeout=t)


def _visit_results(lab, qtbot):
    lab.nav.setCurrentRow(3)
    counts = {}
    for i in range(lab.tabs.count()):
        lab.tabs.setCurrentIndex(i)
        qtbot.wait(150)
        qtbot.waitUntil(lambda: not lab._busy_keys, timeout=60000)
        lab.tabs.currentWidget().grab()
        counts[lab.tabs.tabText(i)] = len(lab.tabs.currentWidget().findChildren(object))
    return counts


@pytest.mark.parametrize("with_video", [False, True])
def test_lab_testset_models_run_results(lab, qtbot, tmp_path, with_video):
    ts = tmp_path / "ts"
    shutil.copytree(f"{D}/ts", ts, ignore=shutil.ignore_patterns("cache"))
    lab._load_data(str(ts))
    _wait_idle(lab, qtbot)
    assert lab.data is not None and lab.data.dataset.items
    lab._add_models([dict(path=f"{D}/models/dummy_a.onnx"), dict(path=f"{D}/models/dummy_b.onnx")])
    if with_video:
        lab.add_video(f"{D}/a.mp4")
    lab.nav.setCurrentRow(2)
    lab.unreviewed.setChecked(True)
    _wait_idle(lab, qtbot)
    lab._run_all()
    _wait_idle(lab, qtbot)
    assert lab.run is not None
    counts = _visit_results(lab, qtbot)
    print(counts)
    # threshold change re-renders
    lab.thr_best.setChecked(False)
    lab.thr_val.setValue(0.4)
    qtbot.wait(300)
    _visit_results(lab, qtbot)
    # report: every kind
    lab.nav.setCurrentRow(4)
    lab.rep_dir.setText(str(tmp_path / "rep"))
    for k, b in lab.rep_btns.items():
        if not b.isEnabled():
            continue
        b.setChecked(True)
        if k == "comparison":
            lab.rep_a.setCurrentIndex(0)
            lab.rep_b.setCurrentIndex(1)
        lab._make_report()
        _wait_idle(lab, qtbot)
        assert "Saved" in lab.rep_status.text(), (k, lab.rep_status.text())
    # project save + reopen
    from PyQt6.QtWidgets import QFileDialog
    import unittest.mock as um
    p = str(tmp_path / "proj.fxlab")
    with um.patch.object(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (p, ""))):
        assert lab._save_project(ask=True)
    assert os.path.isfile(p)
    lab._open_project(p)
    _wait_idle(lab, qtbot)
    qtbot.wait(500)
    _wait_idle(lab, qtbot)


def test_lab_build_testset_from_videos(lab, qtbot, tmp_path):
    from PyQt6.QtWidgets import QFileDialog
    import unittest.mock as um
    lab.kind_ts.setChecked(True)
    lab._add_src_rows([("video", f"{D}/a.mp4"), ("video", f"{D}/b.mp4"), ("folder", f"{D}/ts/images")])
    lab.src_table.item(0, 2).setText("1-6")
    (tmp_path / "newts").mkdir()
    with um.patch.object(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path / "newts"))):
        lab._build_testset()
        _wait_idle(lab, qtbot)
        lab._build_testset()
    _wait_idle(lab, qtbot)
    assert lab.testset is not None and lab.data is not None, lab.sum_lbl.text()
    lab._add_models([dict(path=f"{D}/models/dummy_a.onnx")])
    lab._prelabel()
    _wait_idle(lab, qtbot)


def test_lab_videos_only(lab, qtbot, tmp_path):
    lab.kind_none.setChecked(True)
    lab.add_video(f"{D}/a.mp4")
    lab.add_video(f"{D}/b.mp4", 1.0, 3.0)
    lab._add_models([dict(path=f"{D}/models/dummy_a.onnx"), dict(path=f"{D}/models/dummy_b.onnx")])
    lab.nav.setCurrentRow(2)
    _wait_idle(lab, qtbot)
    lab._run_all()
    _wait_idle(lab, qtbot)
    assert lab.videos_out, (lab.log.toPlainText(), lab.status.currentMessage(), lab.job, len(lab.specs),
                            lab.analysis_videos(), lab.run_btn.text(), lab.m_table.rowCount())
    counts = _visit_results(lab, qtbot)
    print(counts)


def test_lab_editor_ab_plots(lab, qtbot, tmp_path):
    from AIO_QC.widgets.testing.annot_view import ABViewer, LabelEditor, PlotViewer
    from AIO_QC.core.testing.pipeline import frame_getter
    ts = tmp_path / "ts"
    shutil.copytree(f"{D}/ts", ts, ignore=shutil.ignore_patterns("cache"))
    lab._load_data(str(ts))
    _wait_idle(lab, qtbot)
    lab._add_models([dict(path=f"{D}/models/dummy_a.onnx"), dict(path=f"{D}/models/dummy_b.onnx")])
    lab.unreviewed.setChecked(True)
    _wait_idle(lab, qtbot)
    lab._run_all(quiet=True)
    _wait_idle(lab, qtbot)
    r = lab.run
    assert r is not None and r.scored
    items = list(r.eval_idx)[:5]
    changed = []
    ed = LabelEditor(lab, r, lab.data, items, 0, 1, frame_getter, lab._run_slots(r))
    qtbot.addWidget(ed)
    ed.labels_changed.connect(changed.append)
    ed.show()
    n0 = len(ed.cls)
    ed.view.sel = ("pred", 0)
    ed._accept_pred()
    assert len(ed.cls) == n0 + 1
    ed.view.sel = ("gt", 0)
    ed._delete()
    assert ed.save()
    ed._excl(True)
    ed.go(1)
    ed._excl(False)
    ed.done(0)
    assert changed and changed[0]
    lab._labels_changed(changed[0])
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    ab = ABViewer(lab, r, items, 0, 1, 0, frame_getter, lab._run_slots(r), plotter=lab._plotter(),
                  can_edit=lab._can_edit)
    qtbot.addWidget(ab)
    ab.show()
    for i in range(len(items)):
        ab.go(i)
    ab.close()
    pv = PlotViewer(lab, r, items, 0, 0, frame_getter, lab._run_slots(r), plotter=lab._plotter())
    qtbot.addWidget(pv)
    pv.show()
    for i in range(len(items)):
        pv.go(i)
    pv.close()
    # the failure gallery and plots tab fill lazily
    lab.nav.setCurrentRow(3)
    for i in range(lab.tabs.count()):
        lab.tabs.setCurrentIndex(i)
        qtbot.wait(100)
        qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    assert lab._failure_items() or lab._f_rows is not None
