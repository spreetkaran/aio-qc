# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Classifier models on the real inference server with the toy Ultralytics (tests/infer_fake.py; its classifier
calls every 4th image the other class): Model Check finds those as likely mislabels, per-class scores and the
confusion matrix, "Use the model's class" with Undo; Model Assist shows the top classes and sets one."""
import sys
from pathlib import Path

import pytest

from . import fixtures as FX
from .infer_fake import fake_python
from .test_aio_image import shell  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Python is a shell script")


def _setup(root: Path, tmp: Path):
    from AIO_QC.core.state import save_models_config
    m = tmp / "toy_cls.pt"
    m.write_bytes(b"cls")
    save_models_config(root, {"version": 1, "models": [
        {"id": "mc", "name": "toy cls", "path": str(m), "enabled": True, "imgsz": 64, "task": "classify",
         "names": {"0": "cat", "1": "dog", "2": "bird"}, "class_map": {"0": "cat", "1": "dog", "2": ""}}]})


def test_metrics_core():
    from AIO_QC.core import cls_model_check as CM
    recs = [{"key": "a", "split": "", "gt": ["cat"], "pred": [("cat", 0.9), ("dog", 0.1)]},
            {"key": "b", "split": "", "gt": ["cat"], "pred": [("dog", 0.95), ("cat", 0.05)]},
            {"key": "c", "split": "", "gt": ["dog"], "pred": [("dog", 0.4), ("cat", 0.35)]},
            {"key": "d", "split": "", "gt": [], "pred": [("cat", 0.7)]},
            {"key": "e", "split": "", "gt": ["dog"], "pred": [("cat", 0.6), ("dog", 0.4)]}]
    f = {x["key"]: x["kind"] for x in CM.findings(recs)}
    assert f == {"b": CM.LIKELY, "c": CM.UNSURE, "d": CM.NO_CLASS, "e": CM.WRONG}
    m = CM.metrics(recs)
    assert m["n"] == 4 and m["top1"] == 0.5 and m["confusion"]["cat"] == {"cat": 1, "dog": 1}
    assert m["classes"]["cat"]["precision"] == 0.5 and m["classes"]["dog"]["recall"] == 0.5
    assert CM.map_probs([[0, 0.6], [2, 0.3], [1, 0.1]], {"0": "cat", "1": "dog", "2": ""}, {}) == \
        [("cat", 0.6), ("dog", 0.1)]


def test_model_check_and_assist(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.widgets.cls_model_check_dialog import ClsModelCheckDialog, ConfusionDialog
    root = Path(FX.classification(tmp_path / "c", n=6))
    _setup(root, tmp_path)
    w = shell.window("image")
    w.open_dataset(root, D.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    w.inference.set_python(fake_python(tmp_path))
    assert w.qc_actions["model_check"].isEnabled()
    w.open_tool("model_check")
    d = w.open_dialogs["model_check"]
    assert isinstance(d, ClsModelCheckDialog)
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=90000)
    likely = sorted(Path(r.key).name for r in d.rows if r and r.cells[0] == "Likely mislabel")
    assert likely == ["cat_0.jpg", "cat_4.jpg", "dog_0.jpg", "dog_4.jpg"], [r.cells for r in d.rows if r]
    assert "accuracy 66.7%" in d.summarize()
    cls_rows = [r for r in d.rows if r and r.cells[0] == "Class"]
    assert {r.cells[1] for r in cls_rows} == {"cat", "dog"}
    cm = ConfusionDialog(d, d.ctx["metrics"]["confusion"])
    assert cm.table.item(0, 1).text() == "2"                      # 2 cats called dogs
    cm.close()
    # Use the model's class on one finding — and Undo
    row = next(r for r in d.rows if r and Path(r.key).name == "cat_0.jpg")
    d.table.selectRow(d.rows.index(row))
    d.use_model()
    s = w.session
    assert (root / "dog" / "cat_0.jpg").is_file()
    s.undo_class()
    assert (root / "cat" / "cat_0.jpg").is_file()
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=90000)
    assert "from the cache" in d.summarize() and d.ctx["ran"] == 0
    d.close()
    # Model Assist: the top classes on the image shown; one click sets it
    key = str(root / "dog" / "dog_1.jpg")
    w.file_dock.select_key(key)
    qtbot.waitUntil(lambda: w.current_key == key, timeout=5000)
    w.assist.suggest()
    qtbot.waitUntil(lambda: bool(w.assist.items), timeout=30000)
    assert w.assist.items[0]["cls"] == "dog" and not w.assist._cls_bar.isHidden()
    w.assist.accept(1)                                             # the second: cat
    assert (root / "cat" / "dog_1.jpg").is_file()
    s.undo_class()
    w.inference.stop()
