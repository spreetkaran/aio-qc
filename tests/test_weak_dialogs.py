# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Direct tests for the dialogs coverage showed were barely exercised: Crop Gallery (collect, sort, select, relabel,
delete, flag), the OCR validator editor (pattern, learn from examples, code, test, save / copy / delete) and Data
Studio's step form (every kind of field, typed filter values, formula preview, checks)."""
import json

import numpy as np
import pytest
from PyQt6.QtCore import QItemSelectionModel
from PyQt6.QtWidgets import QInputDialog, QWidget

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


# ====================================================================================================== Crop Gallery
def _gallery(w, qtbot, label="car"):
    w.open_tool("crop_gallery")
    g = w.open_dialogs["crop_gallery"]
    i = g.label_box.findData(label)
    assert i >= 0, [g.label_box.itemData(k) for k in range(g.label_box.count())]
    g.label_box.setCurrentIndex(i)
    g.load()
    qtbot.waitUntil(lambda: g.worker is None and g.model.rowCount() > 0, timeout=20000)
    return g


def _select_rows(g, rows):
    sm = g.view.selectionModel()
    sm.clearSelection()
    for r in rows:
        sm.select(g.model.index(r), QItemSelectionModel.SelectionFlag.Select)


def _count(root, label):
    return sum(1 for p in root.glob("*.json") for s in json.loads(p.read_text())["shapes"] if s["label"] == label)


def test_crop_gallery_collects_sorts_and_edits(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    root = FX.labelme(tmp_path / "ds")
    cars = _count(root, "car")
    w = _open(shell, qtbot, root)
    g = _gallery(w, qtbot)
    assert g.model.rowCount() == cars and f"{cars}" in g.status.text()
    # every sort gives the same instances, in some order
    before = sorted(g.model.ids().tolist())
    for k in range(g.sort_box.count()):
        g.sort_box.setCurrentIndex(k)
        qtbot.wait(20)
        assert sorted(g.model.ids().tolist()) == before, g.sort_box.currentText()
    # tiles render (crops are decoded for what is on screen)
    g.show()
    pm = g.pixmap_for(int(g.model.ids()[0]))
    assert pm is not None
    # selection → relabel two cars to "truck"
    _select_rows(g, [0, 1])
    assert len(g.selected_ids()) == 2 and len(g.selected()) == 2
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("truck", True))
    g.relabel()
    qtbot.waitUntil(lambda: _count(root, "truck") == 2, timeout=10000)
    assert _count(root, "car") == cars - 2
    # flag the files of a selection for review
    _select_rows(g, [0])
    g.flag_files()
    assert "Flagged 1 file" in w.statusBar().currentMessage()
    # delete one (the question is answered Yes by the test harness)
    g2 = _gallery(w, qtbot, "truck")
    _select_rows(g2, [0])
    g2.delete()
    qtbot.waitUntil(lambda: _count(root, "truck") == 1, timeout=10000)
    # nothing selected: the actions do nothing
    g2.view.selectionModel().clearSelection()
    assert len(g2.selected_ids()) == 0
    g2.relabel()
    g2.delete()
    assert _count(root, "truck") == 1
    g2.close()


def test_crop_gallery_on_a_list_of_instances(shell, qtbot, tmp_path):  # noqa: F811
    """Opened on explicit (file, shape) pairs — e.g. a confusion-matrix cell."""
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    w.open_tool("crop_gallery")
    g = w.open_dialogs["crop_gallery"]
    keys = [s.key for s in w.session.samples][:3]
    g.show_instances([(k, 0) for k in keys], "Three first shapes")
    qtbot.waitUntil(lambda: g.model.rowCount() == 3, timeout=20000)
    assert g.instance_mode and "Three first shapes" in g.windowTitle()
    assert g.finding_count() == 3 and g.finding_index(keys[1], 0) >= 0
    g.close()


# ====================================================================================================== validators
@pytest.fixture
def editor(tmp_path, qtbot):
    from AIO_QC.widgets.validator_editor import ValidatorEditor
    parent = QWidget()
    reads = {"plate": ["KA01AB1234", "KA01AB1234", "MH12CD5678", "XX", "KA05EF9999", ""]}
    ed = ValidatorEditor(parent, tmp_path, reads_provider=lambda c: reads.get(c, []), class_names=["plate"])
    qtbot.addWidget(parent)
    yield ed
    ed.close()


def test_validator_from_pattern_learned_and_tested(editor, tmp_path):
    ed = editor
    assert ed.current is None and ed.save_btn.isEnabled() and not ed.delete_btn.isEnabled()
    assert any(ed.listw.item(i).text().strip() for i in range(ed.listw.count()))      # the built-in library
    # learn a pattern from the dataset's reads
    ed._examples_from_data()
    assert "KA01AB1234" in ed.examples.toPlainText()
    ed.examples.setPlainText("KA01AB1234\nMH12CD5678\nKA05EF9999")
    ed._learn()
    assert ed.template.text() and "Learned from 3" in ed.learn_out.text()
    ed.name.setText("Plate")
    ed._from_template()
    assert "regex" in ed.regex_out.text() and "def validate" in ed.code.toPlainText()
    # live test as you type
    ed.test_in.setText("KA01AB1234")
    ed._live_test()
    assert "✓ valid" in ed.test_out.text() and "compiles" in ed.code_err.text()
    ed.test_in.setText("hello")
    ed._live_test()
    assert "✗ invalid" in ed.test_out.text()
    # over the dataset's reads
    ed._dataset_test()
    out = ed.ds_out.toPlainText()
    assert "4 / 5 reads pass" in out and "XX" in out
    # save for this dataset, then copy, then delete
    ed.where.setCurrentIndex(ed.where.findData("dataset"))
    ed._save()
    p = tmp_path / ".aio_qc" / "validators" / "plate.py"
    assert p.is_file() and ed.saved_name == "plate" and "NAME = 'Plate'" in p.read_text()
    assert ed.current is not None and ed.current.name == "plate" and ed.delete_btn.isEnabled()   # shown after Save
    ed._copy()
    assert ed.current is None and ed.name.text() == "Plate (copy)" and "Plate (copy)" in ed.code.toPlainText()
    ed._select(next(ed.listw.item(i) for i in range(ed.listw.count()) if ed.listw.item(i).data(0x0100) == "plate"))
    ed._delete()
    assert not p.exists() and ed.current is None


def test_validator_code_errors_and_builtins(editor):
    ed = editor
    ed._new()
    ed.code.setPlainText("def validate(text):\n    return (")             # does not compile
    ed._live_test()
    assert "✗" in ed.code_err.text()
    ed._dataset_test()
    assert "Fix the code first" in ed.ds_out.toPlainText()
    ed.name.setText("")
    ed._save()                                                             # no name: asked for one
    # a built-in one is read-only and cannot be deleted
    builtin = next(ed.listw.item(i) for i in range(ed.listw.count())
                   if ed.listw.item(i).data(0x0100) and ed.vs[ed.listw.item(i).data(0x0100)].origin == "built-in")
    ed.listw.setCurrentItem(builtin)
    assert ed.code.isReadOnly() and not ed.save_btn.isEnabled() and not ed.delete_btn.isEnabled()
    ed._delete()
    assert ed.current is not None and ed.current.path.is_file()
    # an empty template clears the regex line
    ed._new()
    ed.template.setText("")
    ed._from_template()
    assert ed.regex_out.text() == ""


# ====================================================================================================== step form
@pytest.fixture
def df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"name": ["a", "b", "c", "d"], "price": [1.5, 2.0, 3.0, np.nan], "qty": [1, 2, 3, 4],
                         "ok": [True, False, True, True]})


def test_step_form_every_field_kind(df, qtbot):
    from AIO_QC.core.tabular import steps as S
    from AIO_QC.widgets.studio.step_form import AggsEditor, ColumnsList, StepDialog
    parent = QWidget()
    qtbot.addWidget(parent)
    for key, op in S.OPS.items():                                          # every step's form builds and reads back
        d = StepDialog(parent, key, df, tables=["other"])
        try:
            p = d.params()
        except ValueError:
            p = None
        assert p is None or set(p) >= {prm.key for prm in op.params}, key
        d.close()
    # a columns list: filter, all / none
    cl = ColumnsList(["name", "price", "qty"], ["qty"])
    assert cl.value() == ["qty"]
    cl._all(True)
    assert cl.value() == ["name", "price", "qty"]
    cl._filter("pr")
    cl._all(False)
    assert "price" not in cl.value()
    # aggregations
    ag = AggsEditor(["price", "qty"], [("price", "sum")])
    ag.add("qty", "mean")
    assert len(ag.value()) == 2
    ag._remove(ag.rows[0])
    assert len(ag.value()) == 1


def test_step_form_filter_values_are_typed(df, qtbot):
    from AIO_QC.widgets.studio.step_form import StepDialog
    parent = QWidget()
    qtbot.addWidget(parent)
    d = StepDialog(parent, "filter", df)
    d.w["col"].setCurrentText("price")
    d.w["cmp"].setCurrentText(">")
    d.w["value"].setText("2")
    assert d.params()["value"] == 2 and isinstance(d.params()["value"], int)
    d.w["cmp"].setCurrentText("between")
    assert d.w["value2"].isEnabled()
    d.w["value2"].setText("2.5")
    assert d.params()["value2"] == 2.5
    d.w["cmp"].setCurrentText("is one of")
    d.w["value"].setText("1, 2.5")
    assert d.params()["value"] == [1, 2.5]
    d.w["cmp"].setCurrentText("is missing")
    assert not d.w["value"].isEnabled()
    d.w["cmp"].setCurrentText(">")
    d.w["value"].setText("cheap")                                          # not a number for a number column
    d._ok()
    assert "not a number" in d.err.text() and d.result() == 0
    d.w["col"].setCurrentText("ok")
    d.w["cmp"].setCurrentText("=")
    d.w["value"].setText("yes")
    assert d.params()["value"] is True
    d.w["col"].setCurrentText("name")
    d.w["value"].setText("b")
    d._ok()
    assert d.result_params["value"] == "b"


def test_step_form_formula_preview_and_checks(df, qtbot):
    from AIO_QC.widgets.studio.step_form import StepDialog
    parent = QWidget()
    qtbot.addWidget(parent)
    d = StepDialog(parent, "derive", df)
    d.w["formula"].setText("price * qty")
    assert d.preview.text().startswith("First rows → 1.5, 4.0, 9.0")
    d.w["formula"].setText("price * (")
    assert d.preview.text().startswith("⚠")
    d.w["formula"].setText("")
    assert d.preview.text() == ""
    d.w["formula"].setText("price * qty")
    d.w["name"].setText("")
    d._ok()
    assert "Name" in d.err.text()                                          # the step's own check
    d.w["name"].setText("total")
    d._ok()
    assert d.result_params["name"] == "total" and "text_cols" in d.result_params
    r = StepDialog(parent, "replace", df)
    r.w["col"].setCurrentText("qty")
    r.w["old"].setText("3")
    r.w["new"].setText("30")
    p = r.params()
    assert p["old"] == 3 and p["new"] == 30
