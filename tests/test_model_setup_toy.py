# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Models & Environment on the real inference server with the toy Ultralytics (tests/infer_fake.py): check an
environment, add models, load them (classes read, mapped to the dataset's labels by name), edit a mapping, image
size, test on the shown image, export, remove — and the install plan asked about, not run."""
import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from . import fixtures as FX
from .infer_fake import fake_python
from .test_aio_image import _open, shell  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Python is a shell script")


def test_models_and_environment(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core.state import load_models_config
    from AIO_QC.widgets.model_setup_dialog import ClassMapDialog, auto_class_map
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    qtbot.waitUntil(lambda: w.current_key is not None, timeout=10000)
    py = fake_python(tmp_path)
    w.open_model_setup()
    d = w.open_dialogs["model_setup"]
    # 1 · the environment: probed for real (the toy torch / ultralytics)
    d.check_env(py)
    qtbot.waitUntil(lambda: w.inference.info is not None, timeout=60000)
    assert w.inference.python == py and w.inference.info.get("ultralytics") == "8.3.0-toy"
    d._refresh_env_status()
    d._fill_devices()
    d._on_device()
    # 2 · add two models (one detects cars and plates, one outlines cars)
    det, seg = tmp_path / "toy_det.pt", tmp_path / "toy_seg.pt"
    det.write_bytes(b"det")
    seg.write_bytes(b"seg")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(det), str(seg)], ""))
    d.add_models()
    d.add_models()                                                   # the same files again: not added twice
    assert [Path(m["path"]).name for m in d.models] == ["toy_det.pt", "toy_seg.pt"]
    # 3 · load: classes and task come from the models, mapped to the dataset's labels by name
    d.load_models()
    qtbot.waitUntil(lambda: all(m.get("names") for m in d.models), timeout=60000)
    m0, m1 = d.models
    assert m0["names"] == {"0": "car", "1": "plate"} and m0["class_map"] == {"0": "car", "1": "plate"}
    assert m1["task"] == "segment" and m1["class_map"] == {"0": "car"}
    assert load_models_config(w.session.root)["models"][0]["class_map"]["1"] == "plate"
    # mapping by hand
    cm = ClassMapDialog(d, m0, d._labels(), w.session.adapter.map_label)
    cm._auto()
    assert cm.result_map() == {"0": "car", "1": "plate"}
    cm.close()
    assert auto_class_map({"0": "Car", "1": "bike"}, ["car"]) == {"0": "car", "1": ""}
    d._set_imgsz(m0, "256")
    assert d.models[0]["imgsz"] == 256
    d._refresh_coverage()
    # 4 · test on the image shown: dashed boxes on the canvas
    d.test_current()
    qtbot.waitUntil(lambda: "after mapping" in d.console.toPlainText(), timeout=60000)
    # 5 · export the selected model (the toy writes a file; it is kept with the dataset)
    d.table.selectRow(0)
    d.export_model("onnx")
    qtbot.waitUntil(lambda: any((w.session.root / ".aio_qc" / "exports").glob("*.onnx"))
                    if (w.session.root / ".aio_qc" / "exports").exists() else False, timeout=60000)
    # 6 · an install is planned and asked about first (answered No here: nothing runs)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    d.start_install()
    assert d.proc is None
    # 7 · remove one
    d.table.selectRow(1)
    d.remove_model()
    assert [Path(m["path"]).name for m in d.models] == ["toy_det.pt"]
    d.close()
    w.inference.stop()
