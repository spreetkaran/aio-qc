# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Model Lab with classifiers, on real Qt and real ONNX models: the handwritten-digits dataset as class folders (with
planted mislabels), two exported classifiers (a logistic regression each, metadata task=classify). Checks the run is
scored as classification (top-1 / top-5 / F1), every results tab shows classifier content, the confidence threshold
gives "answered / right when sure", confusion cells filter the mistakes, an image's class can be changed from the
viewer (the file moves to the other class folder and the run re-scores), reports and exports, project round trip."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from .test_fx_lab import lab, _wait_idle, _visit_results  # noqa: F401  (fixture)
from .test_modes import shell3  # noqa: F401  (fixture)

pytest.importorskip("sklearn")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture(autouse=True)
def _keep_lab_session():
    """The Lab restores its last session (models, data): these tests start from an empty one and leave the
    session the other tests expect as they found it."""
    from AIO_QC.core.testing import project as PJ
    saved = {}
    for p in (PJ.AUTOSAVE, PJ._RECENT):
        saved[p] = open(p, "rb").read() if os.path.isfile(p) else None
    yield
    for p, data in saved.items():
        if data is None:
            if os.path.isfile(p):
                os.remove(p)
        else:
            with open(p, "wb") as f:
                f.write(data)


def _fresh(lab, qtbot):
    lab._apply_state(dict(name="Untitled"))
    _wait_idle(lab, qtbot)
    assert lab.m_table.rowCount() == 0


@pytest.fixture
def digits(tmp_path):
    import cv2
    from .digits_dataset import build, onnx_classifier
    root = tmp_path / "digits"
    meta = build(root)
    imgs = [cv2.imread(p) for p in meta["paths"]]
    truth = [meta["truth"][p]["true"] for p in meta["paths"]]
    a = onnx_classifier(tmp_path / "digits-a.onnx", imgs, truth, meta["names"], c=1.0)
    b = onnx_classifier(tmp_path / "digits-b.onnx", imgs, truth, meta["names"], c=0.002)
    return root, meta, a, b


def test_lab_classifiers(lab, qtbot, tmp_path, digits):  # noqa: F811
    root, meta, ma, mb = digits
    _fresh(lab, qtbot)
    lab._load_data(str(root))
    _wait_idle(lab, qtbot)
    ds = lab.data.dataset
    assert ds.task == "classify" and len(ds.items) == len(meta["paths"]) and ds.cls_type is not None
    lab._add_models([dict(path=str(ma)), dict(path=str(mb))])
    lab.nav.setCurrentRow(2)
    _wait_idle(lab, qtbot)
    lab._run_all()
    _wait_idle(lab, qtbot)
    r = lab.run
    assert type(r).__name__ == "ClsLabRun", lab.log.toPlainText()[-800:]
    rows = {x["model"]: x for x in r.summary_table()}
    a, b = rows["digits-a"], rows["digits-b"]
    assert a["top1"] > 0.85 and a["top5"] > a["top1"] and b["top1"] < a["top1"], rows
    assert a["rank"] == 1
    counts = _visit_results(lab, qtbot)
    assert "Mistakes" in counts and "Answers" in counts, counts
    hdr = [lab.board.horizontalHeaderItem(i).text() for i in range(lab.board.columnCount())]
    assert "Top-1" in hdr and "mAP50-95" not in hdr
    assert lab.board.item(0, 3).text().endswith("%")
    assert lab.pc_table.horizontalHeaderItem(8).text() == "Mostly confused with"
    # charts: accuracy-vs-coverage, per-class F1, ...; confusion matrix by class name
    lab.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    assert lab.charts.cards()
    # mistakes: the planted mislabels are the surest ones
    from AIO_QC.widgets.testing.model_lab_common import T_FAILURES
    lab.tabs.setCurrentIndex(T_FAILURES)
    lab.f_kind.setCurrentIndex(lab.f_kind.findData("sure"))
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    assert lab.f_list.count() > 0 and "→" in lab.f_list.item(0).text()
    wrong = {Path(p).stem for p in meta["wrong"]} | {Path(p).stem for p in meta["dups"]}
    top = [Path(r.items[ii].path).stem for ii in lab._failure_items()[:20]]
    assert sum(1 for t in top if t in wrong) >= 15, top
    # a confusion cell filters the gallery
    t, p = next((g, k) for g, row in r.results[0].confusion.items() for k, n in row.items() if k != g and n)
    lab._on_link(f"cm:0:{t}:{p}")
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    assert lab.f_chip.isVisible() and t in lab.f_chip.text()
    assert all(t in r.items[ii].tags and r.preds[0][ii][0][0] == p for ii in lab._failure_items())
    lab._clear_cm_filter()
    # threshold = "unsure below": fewer answered, those more often right
    lab.res_best.setChecked(False)
    lab.res_slider.setValue(70)
    qtbot.wait(400)
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    ra = r.results[0]
    assert ra.answered < 1.0 and ra.acc_answered > ra.top1
    lab.res_best.setChecked(True)
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    # the viewer: answer under the image; give a mislabelled image its true class (it moves folder, re-scored)
    from AIO_QC.widgets.testing.cls_view import ClsViewer
    ii = next(i for i in lab._failure_items() if Path(r.items[i].path).stem in wrong)
    item = r.items[ii]
    old_path = item.path
    true = meta["truth"][old_path]["true"]
    before = r.results[0].top1
    v = ClsViewer(lab, r, [ii], [0], 0, lab._plotter(), lab._cls_can_edit, lab._cls_set_class)
    qtbot.addWidget(v)
    assert v.apply_b.isVisible() is False or v.cls_box.isVisibleTo(v)
    img = lab._plotter()(ii, 0, True)
    assert img is not None and img.shape[0] > img.shape[1]          # the answer band is under the picture
    changed = []
    v.labels_changed.connect(changed.extend)
    v.cls_box.setCurrentIndex(v.cls_box.findText(true))
    v._apply()
    assert changed == [ii] and item.tags == [true]
    assert not os.path.exists(old_path) and Path(item.path).parent.name == true and os.path.isfile(item.path)
    v.close()
    lab._labels_changed(changed)
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=30000)
    assert r.results[0].top1 > before
    # side by side
    v2 = ClsViewer(lab, r, list(r.eval_idx[:5]), [0, 1], 0, lab._plotter())
    qtbot.addWidget(v2)
    assert len(v2.views) == 2 and not v2.apply_b.isVisibleTo(v2)
    v2.go(3)
    v2.close()
    # reports + exports
    lab.nav.setCurrentRow(4)
    lab.rep_dir.setText(str(tmp_path / "rep"))
    for k, btn in lab.rep_btns.items():
        if not btn.isEnabled():
            continue
        btn.setChecked(True)
        if k == "comparison":
            lab.rep_a.setCurrentIndex(0)
            lab.rep_b.setCurrentIndex(1)
        lab._make_report()
        _wait_idle(lab, qtbot)
        assert "Saved" in lab.rep_status.text(), (k, lab.rep_status.text())
    html = [p for p in (tmp_path / "rep").rglob("*.html")]
    assert html and all("Top-1" in p.read_text(encoding="utf-8") or "Only" in p.read_text(encoding="utf-8")
                        for p in html)
    pi = next((tmp_path / "rep").rglob("per_image.csv"))
    rows = list(csv.DictReader(open(pi, encoding="utf-8")))
    assert rows and {"true", "predicted", "confidence", "correct"} <= set(rows[0])
    # project round trip: the classifier run comes back from the cache
    from PyQt6.QtWidgets import QFileDialog
    import unittest.mock as um
    p = str(tmp_path / "proj.fxlab")
    with um.patch.object(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (p, ""))):
        assert lab._save_project(ask=True)
    lab._open_project(p)
    _wait_idle(lab, qtbot)
    qtbot.wait(500)
    _wait_idle(lab, qtbot)
    assert lab.run is not None and type(lab.run).__name__ == "ClsLabRun" and lab.run.scored


def test_player_draws_a_classifier_answer_as_a_pill(qapp):
    """A classifier's whole-frame rows: one pill with the best class in the corner — no frame-sized rectangle."""
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QColor, QImage, QPainter
    from AIO_QC.tools.frame_extractor.widgets import det_paint
    rows = np.array([[0, 0, 640, 360, 0.2, 3], [0, 0, 640, 360, 0.7, 1]], np.float32)
    best = det_paint.classifier_answer(rows)
    assert best is not None and int(best[5]) == 1
    assert det_paint.classifier_answer(np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)) is None
    img = QImage(640, 360, QImage.Format.Format_RGB888)
    img.fill(QColor("white"))
    p = QPainter(img)
    det_paint.paint(p, [(0, 0, 640, 360, det_paint.label_text("cat", 0.7), QColor("#3987e5"), False, None, 0.7,
                         "class")], lambda x, y: QPointF(x, y), QRectF(0, 0, 640, 360), 11)
    p.end()
    assert img.pixelColor(20, 18) != QColor("white")                       # the pill
    assert img.pixelColor(320, 359) == QColor("white") and img.pixelColor(639, 180) == QColor("white")  # no frame


def test_lab_classifiers_on_video(lab, qtbot, tmp_path, digits):  # noqa: F811
    """Classifiers on a video: the answer per sampled frame, brief flips / drop-outs, agreement; the Videos tab and
    the video report use classifier wording."""
    from .fx_fixtures import D
    root, meta, ma, mb = digits
    _fresh(lab, qtbot)
    lab.kind_none.setChecked(True)
    lab._add_models([dict(path=str(ma)), dict(path=str(mb))])
    lab.add_video(f"{D}/a.mp4")
    lab.nav.setCurrentRow(2)
    _wait_idle(lab, qtbot)
    lab._run_all()
    _wait_idle(lab, qtbot)
    assert lab.videos_out and lab.videos_out[0][1]["task"] == "classify", lab.log.toPlainText()[-600:]
    v = lab.videos_out[0][1]
    assert v["models"][0]["frames_with_dets"] == len(v["times"])     # a classifier answers every frame
    from AIO_QC.widgets.testing.model_lab_common import T_VIDEOS
    lab.nav.setCurrentRow(3)
    lab.tabs.setCurrentIndex(T_VIDEOS)
    qtbot.waitUntil(lambda: not lab._busy_keys, timeout=60000)
    assert lab.v_grid.cards()
    from AIO_QC.core.testing import pipeline as P
    out = P.make_report(str(tmp_path / "v.html"), "video", lab.run, lab.videos_out, slots=[0, 1])
    html = open(out[0], encoding="utf-8").read()
    assert "Confidence over time" in html and "Brief flips" in html


def test_testing_on_a_classification_dataset_in_image_qc(shell3, qtbot, tmp_path, digits):  # noqa: F811
    """Image QC ▸ Testing on an open class-folder dataset: scored as classification, and a class changed from the
    Testing viewer goes through Image QC (the image moves, File History, one step of Undo)."""
    from AIO_QC.core import model_registry as MR
    from AIO_QC.core.dataset import DatasetType
    root, meta, ma, mb = digits
    sh = shell3
    sh.show("image")
    w = sh.window("image")
    w.open_dataset(root, DatasetType.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=60000)
    MR.register(str(ma), w.session.root)
    lab = w.open_testing()
    qtbot.addWidget(lab)
    qtbot.waitUntil(lambda: lab.data is not None and lab.job is None, timeout=60000)
    _wait_idle(lab, qtbot)
    assert lab.data.qc and lab.data.dataset.task == "classify" and lab.data.dataset.cls_type is not None
    lab._run_all(quiet=True)
    _wait_idle(lab, qtbot)
    r = lab.run
    assert type(r).__name__ == "ClsLabRun" and r.results[0].top1 > 0.85, lab.log.toPlainText()[-600:]
    wrong = {Path(p).stem: meta["truth"][p]["true"] for p in meta["wrong"]}
    ii = next(i for i, it in enumerate(r.items) if Path(it.path).stem in wrong)
    item = r.items[ii]
    old = item.path
    lab._cls_set_class(ii, wrong[Path(old).stem])
    assert not os.path.exists(old) and Path(item.path).parent.name == wrong[Path(old).stem]
    s = w.session
    k = s.key_by_path.get(item.path) or next(k for k, smp in s.by_key.items() if str(smp.image_path) == item.path)
    assert s.tags_of(k) == [wrong[Path(old).stem]]
    lab.close()

