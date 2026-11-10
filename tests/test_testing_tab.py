# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Testing (Image QC ▸ Models): on the open dataset, with AIO QC's models, label fixes through AIO QC, exclusions with
the dataset, the project as part of the dataset's session, reached from the Frame Extractor without its player."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from . import fixtures as FX
from .fx_fixtures import D
from .test_aio_image import _open
from .test_modes import shell3  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    from AIO_QC.widgets.testing import lab_common, model_lab
    monkeypatch.setattr(lab_common, "open_path", lambda *a, **k: None)
    monkeypatch.setattr(model_lab, "open_path", lambda *a, **k: None)


def _idle(lab, qtbot, t=120000):
    qtbot.wait(20)                                   # deferred restore / loads start on the next event loop turn
    qtbot.waitUntil(lambda: lab.job is None and not lab._busy_keys, timeout=t)


def _opened(lab, qtbot):
    qtbot.waitUntil(lambda: lab.data is not None and lab.job is None, timeout=60000)
    _idle(lab, qtbot)


def _models_in_dataset(root):
    from AIO_QC.core import model_registry as R
    for name in ("dummy_a.onnx", "dummy_b.onnx"):
        R.register(os.path.join(D, "models", name), root)


@pytest.mark.parametrize("kind", ["labelme", "yolo"])
def test_testing_runs_on_the_qc_dataset(shell3, qtbot, tmp_path, kind):  # noqa: F811
    sh = shell3
    sh.show("image")
    root = (FX.labelme if kind == "labelme" else FX.yolo)(tmp_path / kind)
    w = _open(sh, qtbot, root)
    _models_in_dataset(w.session.root)
    lab = w.open_testing()
    qtbot.addWidget(lab)
    _opened(lab, qtbot)
    assert lab.kind_qc.isChecked(), "the open dataset is the default test data"
    assert lab.data is not None and lab.data.qc and len(lab.data.dataset.items) >= 10
    assert lab.data.dataset.n_labelled >= 10
    assert lab.m_table.rowCount() == 2, "the dataset's models are the default models"
    lab.unreviewed.setChecked(True)
    lab.nav.setCurrentRow(2)
    _idle(lab, qtbot)
    lab._run_all()
    _idle(lab, qtbot)
    assert lab.run is not None and lab.run.scored, lab.log.toPlainText()[-800:]
    lab.nav.setCurrentRow(3)
    empty = []
    for i in range(lab.tabs.count()):
        lab.tabs.setCurrentIndex(i)
        qtbot.wait(100)
        _idle(lab, qtbot)
        page = lab.tabs.currentWidget()
        page.grab()
        visible = [c for c in page.findChildren(object) if getattr(c, "isVisible", lambda: False)()]
        if len(visible) < 5:
            empty.append(lab.tabs.tabText(i))
    assert not empty, f"sub-tabs without content: {empty}"
    # one prediction cache with the dataset (Model Check reads it)
    from AIO_QC.core.model_registry import shared_cache_path
    assert shared_cache_path(w.session.root).is_file()
    # the Testing project is part of the dataset's session
    lab._autosave()
    from AIO_QC.core.testing import project as PJ
    p = PJ.dataset_project(w.session.root)
    assert os.path.isfile(p) and json.load(open(p))["data"]["kind"] == "qc"


def test_label_fix_and_exclusion_go_through_aio_qc(shell3, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets.testing.annot_view import LabelEditor
    from AIO_QC.core.testing.pipeline import frame_getter
    from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG
    sh = shell3
    sh.show("image")
    w = _open(sh, qtbot, FX.labelme(tmp_path / "ds"))
    _models_in_dataset(w.session.root)
    lab = w.open_testing()
    qtbot.addWidget(lab)
    _opened(lab, qtbot)
    lab.unreviewed.setChecked(True)
    lab._run_all(quiet=True)
    _idle(lab, qtbot)
    r = lab.run
    ii = list(r.eval_idx)[0]
    item = r.items[ii]
    key = w.session.key_by_path[str(item.label_path)]
    w.session.set_review([key], REVIEW_FLAG, "test")
    before = Path(item.label_path).read_bytes()
    w.open_sample(key)
    ed = LabelEditor(lab, r, lab.data, [ii], 0, 0, frame_getter, lab._run_slots(r))
    ed.qc = lab.qc
    qtbot.addWidget(ed)
    ed.view.sel = ("gt", 0)
    ed._delete()
    assert ed.save()
    after = Path(item.label_path).read_bytes()
    assert after != before
    s = w.session
    assert s.review.status(key) == REVIEW_FIXED                     # flagged -> fixed, as any AIO QC fix
    events = s.history.for_file(key)
    assert any(e.get("tool") == "Testing" for e in events), events
    from AIO_QC.core.backups import BackupStore
    ops = BackupStore(s.root).ops()
    assert any(o.get("kind") == "testing" or "Testing" in str(o.get("title", "")) for o in ops), ops
    # leave out of scoring -> stored with the dataset
    ed._excl(True)
    assert (Path(s.root) / ".aio_qc" / "testing_excluded.json").is_file()
    ed.done(0)


def test_testing_is_not_part_of_the_frame_extractor(shell3, qtbot, tmp_path):  # noqa: F811
    sh = shell3
    sh.show("video")
    v = sh.video
    assert not hasattr(v, "open_lab") and not hasattr(v, "lab")
    texts = []
    for a in v.findChildren(__import__("PyQt6.QtGui", fromlist=["QAction"]).QAction):
        texts.append(a.text().replace("&", "").lower())
    assert not any("testing" in t for t in texts), [t for t in texts if "testing" in t]
    # Testing is still reachable from Image QC and never touches the player
    sh.switch("image")
    lab = sh.image.open_testing()
    qtbot.addWidget(lab)
    for name in dir(lab):
        assert "open_at" not in name and "player" not in name.lower()


def test_model_check_reuses_testing_predictions(shell3, qtbot, tmp_path):  # noqa: F811
    """One prediction cache per dataset: after Testing ran a model, Model Check finds its predictions (same
    weights, size and SAHI) instead of running the model again."""
    from AIO_QC.core import model_registry as R
    from AIO_QC.widgets.model_check_dialog import ModelCheckDialog
    sh = shell3
    sh.show("image")
    w = _open(sh, qtbot, FX.labelme(tmp_path / "ds"))
    root = w.session.root
    R.register(os.path.join(D, "models", "dummy_a.onnx"), root, imgsz=320)
    lab = w.open_testing()
    qtbot.addWidget(lab)
    _opened(lab, qtbot)
    lab._run_all(quiet=True)
    _idle(lab, qtbot)
    m = R.dataset_models(root)[0]
    hit = R.shared_lookup(root, m["path"], 320, None)
    assert hit is not None
    smp = next(s for s in w.session.samples if s.ann_path is not None)
    img = w.session.adapter.resolve_image(smp, None) if hasattr(w.session.adapter, "resolve_image") else smp.image_path
    ctx = {"shared": {m["id"]: hit}, "root": root, "classes": {}}
    got = ModelCheckDialog._from_shared(ctx, m["id"], img)
    assert got is not None and got["w"] == 320 and got["boxes"], got
    assert all(b[4] >= 0.05 for b in got["boxes"])
    assert R.shared_lookup(root, m["path"], 640, None) is None          # another size: never mixed
