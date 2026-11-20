# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The classifier versions of the box checks, each on a class-folder dataset with the problem planted: the toolbar
button opens the classifier version, it finds the planted problem, and its fix goes through Image QC."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from .test_aio_image import shell  # noqa: F401


def _img(path, seed, w=160, h=120, colour=None, bright=None):
    rng = np.random.default_rng(seed)
    a = (rng.random((h, w, 3)) * 40 + (bright if bright is not None else 100)).astype(np.uint8)
    if colour is not None:
        a[:] = colour
    cv2.rectangle(a, (20 + seed % 30, 20), (90, 90), (40, 40, 40), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), a)


def _dataset(root: Path):
    """cat / dog frames of one video (frames 10-11 of 'cat' filed as dog), name variants, a copy in another class,
    an odd image, stray files."""
    for i in range(1, 31):
        cls = "dog" if i in (10, 11) else "cat"
        _img(root / cls / f"clip_{i:04d}.png", i)
    for i in range(12):
        _img(root / "dog" / f"dog_{i:03d}.png", 100 + i, bright=60)
    _img(root / "Dog" / "odd_0.png", 300)                         # one class written two ways
    _img(root / "Dog" / "odd_1.png", 301)
    _img(root / "cats" / "plural_0.png", 302)                    # singular / plural
    _img(root / "big Bird-x" / "bb_0.png", 303)                   # mixed separators / case
    shutil.copy(root / "cat" / "clip_0003.png", root / "dog" / "copy_of_clip3.png")   # a copy under another class
    _img(root / "cat" / "huge_bright.png", 400, w=1600, h=300, colour=(30, 230, 250))  # unusual for 'cat'
    _img(root / "cat" / "tiny.png", 401, w=24, h=24)
    (root / "cat" / "notes.pdf").write_bytes(b"%PDF-1.4 not an image")
    (root / "cat" / "clip_0005.json").write_text('{"shapes": [{"label": "dog"}], "flags": {}}')
    (root / "dog" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    _img(root / "cat" / "wrong_ext.png", 500)
    os.replace(root / "cat" / "wrong_ext.png", root / "cat" / "wrong_ext.jpg")    # PNG bytes, .jpg name
    (root / "cat" / "empty.jpg").write_bytes(b"")
    _img(root / "loose.png", 600)                                  # outside every class folder
    (root / "bird").mkdir()                                        # an empty class folder
    return root


@pytest.fixture
def cls_ds(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType
    root = _dataset(tmp_path / "clsds")
    w = shell.window("image")
    w.open_dataset(root, DatasetType.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    return w


def _run(w, qtbot, key):
    from AIO_QC.app_common import QC_TOOLS, tool_for, tool_supported
    t = next(t for t in QC_TOOLS if t.key == key)
    assert tool_supported(t, w.session), key
    assert tool_for(t, w.session).dialog is not t.dialog, f"{key}: the classifier version"
    assert w.qc_actions[key].isEnabled(), key
    w.open_tool(key)
    d = w.open_dialogs[key]
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    if not any(d.rows):
        d.start_scan()
        qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    return d, [r for r in d.rows if r]


def _tick(d, pred):
    from PyQt6.QtCore import Qt
    for r in range(d.table.rowCount()):
        row = d._row_at(r)
        if row is not None:
            d.table.item(r, 0).setCheckState(Qt.CheckState.Checked if pred(row) else Qt.CheckState.Unchecked)


def test_ambiguity_and_name_text(cls_ds, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: (a[3][0], True)))
    w = cls_ds
    d, rows = _run(w, qtbot, "ambiguity")
    by = {(r.cells[0], r.cells[1]) for r in rows}
    assert ("Written two ways", "Dog") in by and ("Singular / plural", "cats") in by, by
    _tick(d, lambda r: r.cells[1] == "Dog")
    d.merge_into()                                                 # QInputDialog.getItem -> the suggestion
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    assert "Dog" not in w.session.cls_classes()
    assert not (Path(w.session.root) / "Dog").exists() or not any((Path(w.session.root) / "Dog").iterdir())
    d.close()
    d, rows = _run(w, qtbot, "text_labels")
    kinds = {r.cells[0] for r in rows if r.cells[1] == "big Bird-x"}
    assert "Mixed separators" in kinds, rows
    d.close()


def test_duplicates_outliers_sequence_sizes(cls_ds, qtbot):
    w = cls_ds
    s = w.session
    d, rows = _run(w, qtbot, "duplicates")
    other = [r for r in rows if str(r.cells[5]).startswith("⚠ OTHER class")]
    assert other and Path(other[0].key).name in ("copy_of_clip3.png", "clip_0003.png")
    _tick(d, lambda r: str(r.cells[5]).startswith("⚠"))
    d.use_kept_class()
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    copies = [k for k in s.by_key if Path(k).name in ("copy_of_clip3.png", "clip_0003.png")]
    assert len({tuple(s.tags_of(k)) for k in copies}) == 1
    d.close()
    d, rows = _run(w, qtbot, "outliers")
    odd = {Path(r.key).name for r in rows}
    assert "huge_bright.png" in odd, odd
    d.close()
    d, rows = _run(w, qtbot, "sequence")
    flips = sorted(Path(r.key).name for r in rows)
    assert flips == ["clip_0010.png", "clip_0011.png"], flips
    _tick(d, lambda r: True)
    d.use_neighbours()
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    assert all(s.tags_of(k) == ["cat"] for k in s.by_key if Path(k).name in ("clip_0010.png", "clip_0011.png"))
    d.close()
    d, rows = _run(w, qtbot, "dim_area")
    assert "tiny.png" in {Path(r.key).name for r in rows}
    assert "huge_bright.png" in {Path(r.key).name for r in rows if r.cells[0].startswith("Shape")}
    d.close()
    d, rows = _run(w, qtbot, "train_size")
    assert any(r.cells[0] == "Enlarged" and Path(r.key).name == "tiny.png" for r in rows)
    assert any(r.cells[0] == "Centre crop cuts much" and Path(r.key).name == "huge_bright.png" for r in rows)
    assert any(r.cells[0] == "Class summary" for r in rows)
    d.close()


def test_validation_and_pairs(cls_ds, qtbot):
    w = cls_ds
    d, rows = _run(w, qtbot, "validation")
    kinds = {(r.cells[0], Path(r.cells[2]).name) for r in rows}
    for want in (("Not in a class folder", "loose.png"), ("Not an image", "notes.pdf"),
                 ("Wrong extension", "wrong_ext.jpg"), ("Empty file", "empty.jpg"),
                 ("Empty class folder", "bird")):
        assert want in kinds, (want, kinds)
    d.close()
    d, rows = _run(w, qtbot, "pairs")
    kinds = {(r.cells[0], Path(r.cells[2]).name) for r in rows}
    assert ("Annotation file next to an image", "clip_0005.json") in kinds, kinds
    assert ("Annotation without an image", "orphan.txt") in kinds, kinds
    d.close()


def test_ocr_and_model_check_open(cls_ds, qtbot):
    """Their classifier versions open (OCR needs an engine from your environment; Model Check a classifier)."""
    from AIO_QC.app_common import QC_TOOLS, tool_for
    w = cls_ds
    for key, name in (("ocr_check", "ClsOcrDialog"), ("model_check", "ClsModelCheckDialog")):
        t = next(t for t in QC_TOOLS if t.key == key)
        assert tool_for(t, w.session).dialog.__name__ == name
        assert w.qc_actions[key].isEnabled()
        w.open_tool(key)
        d = w.open_dialogs[key]
        assert type(d).__name__ == name
        d.close()


def test_ocr_reads_whole_images_against_their_class(cls_ds, qtbot):
    """OCR (a stand-in engine): the text in each whole image is compared with its class name."""
    import sys
    from PyQt6.QtCore import Qt
    w = cls_ds

    class Svc:
        load_info = {}

        def shutdown(self):
            pass

        def ensure_loaded(self, *a, **k):
            return {}

        def read(self, items, pad, min_h, cancel=None):
            out = []
            for it in items:
                p = Path(it["path"])
                txt = "dog" if p.name == "clip_0004.png" else p.parent.name      # a 'cat' image that says DOG
                out.append({"w": 160, "h": 120, "boxes": [{"text": txt, "conf": 0.97}]})
            return out
    real = w.ocr
    w.ocr = Svc()
    w.open_tool("ocr_check")
    d = w.open_dialogs["ocr_check"]
    d.python.setCurrentText(sys.executable)
    d.engine.addItem("EasyOCR (stand-in)", "easyocr")
    d.engine.setCurrentIndex(d.engine.findData("easyocr"))
    for lab, (it, *_r) in d._cls_widgets.items():
        it.setCheckState(Qt.CheckState.Checked if lab in ("cat", "dog") else Qt.CheckState.Unchecked)
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    rows = [r for r in d.rows if r]
    assert [Path(r.key).name for r in rows] == ["clip_0004.png"], [(r.key, r.cells) for r in rows]
    assert "class “CAT”" in rows[0].cells[4] and rows[0].shape_index is None
    d.close()
    w.ocr = real


def test_pick_for_a_classifier(cls_ds, qtbot, tmp_path):
    """Data Prep ▸ Pick with a classification model: unsure = a close call between two classes; the picks arrive
    sorted into class folders by the model's guess (no empty label files)."""
    import threading
    from AIO_QC.core.prep import pick as PK
    from AIO_QC.core.state import save_models_config
    from AIO_QC.widgets.data_prep import PickPage
    assert PK.cls_uncertainty([[0, 0, 9, 9, 0.5, 1], [0, 0, 9, 9, 0.45, 0]]) == pytest.approx((0.95, 1))
    assert PK.cls_uncertainty([[0, 0, 9, 9, 0.99, 2]])[0] < 0.05
    w = cls_ds
    pool = tmp_path / "pool"
    for i in range(12):
        _img(pool / f"p_{i:02d}.png", 900 + i)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    save_models_config(w.session.root, {"models": [{"id": "m1", "path": str(model), "enabled": True,
                                                    "task": "classify"}]})

    class Svc:
        python, opts, info = "py", {"batch": 8}, {}

        def ensure_started(self):
            pass

        def ensure_loaded(self, specs):
            return {specs[0]["id"]: {"names": {0: "cat", 1: "dog"}, "task": "classify"}}

        def predict(self, part, ids, conf=0.05, cancel=None):
            out = {}
            for p in part:
                n = int(Path(p).stem.split("_")[1])
                c = 0.5 + 0.04 * (n % 6)                       # some close calls, some sure
                out[str(p)] = {"by_model": {ids[0]: [[0, 0, 160, 120, c, n % 2], [0, 0, 160, 120, 1 - c,
                                                                                    1 - n % 2]]}}
            return out
    real = w.inference
    w.inference = Svc()
    w.open_data_prep()
    dlg = w.open_dialogs["data_prep"]
    pg = dlg.show_page(PickPage)
    before = {c: c.isChecked() for c in (pg.unsure, pg.unlike, pg.stubs, pg.sort_cls)}     # remembered settings
    pg.src.setText(str(pool))
    pg.out.setText(str(tmp_path / "picked"))
    pg.n.setValue(6)
    pg.unsure.setChecked(True)
    pg.unlike.setChecked(False)
    pg.stubs.setChecked(True)                               # ignored for a classification dataset
    pg.sort_cls.setChecked(True)
    plan = pg.make_plan(threading.Event(), lambda *a: None)
    assert {Path(a[2]).parent.name for a in plan.actions} <= {"cat", "dog"}
    st = pg.run(plan, threading.Event(), lambda *a: None)
    assert st["done"] == 6 and st.get("stubs") is None
    assert not list((tmp_path / "picked").rglob("*.json"))
    for c, v in before.items():
        c.setChecked(v)
    dlg.close()
    w.inference = real


def test_data_studio_table_of_a_classifier_dataset(cls_ds):
    """Data Studio's annotation table: one row per image class, with the image's size and shape."""
    from AIO_QC.core.tabular import annotations as AN
    s = cls_ds.session
    df = AN.build(s.samples, s.base_adapter, s.review)
    rows = df[df["shape_type"] == "image class"]
    assert len(rows) >= 40 and set(rows["label"]) >= {"cat", "dog"}
    big = rows[rows["file"].str.endswith("huge_bright.png")].iloc[0]
    assert big["width"] == 1600 and big["height"] == 300 and abs(big["aspect"] - 1600 / 300) < 1e-3
