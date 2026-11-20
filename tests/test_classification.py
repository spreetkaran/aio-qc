# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Classification datasets: what is detected (train + test without val, stray files, flags that are not all clean,
boxes AND image flags), how flags are read (1 / "true" count, and are reported), single- / multi-label, Change class
with the picker, 1 … 9, Undo / Redo — for LabelMe flags and class folders."""
import json
import os
import shutil
from pathlib import Path

from . import fixtures as FX
from .test_aio_image import shell  # noqa: F401


def lm_flags(root: Path, items):
    root.mkdir(parents=True, exist_ok=True)
    for i, (name, flags, shapes) in enumerate(items):
        img = root / f"{name}.jpg"
        FX._img(str(img), seed=i)
        d = {"version": "5.5.0", "flags": flags, "shapes": shapes, "imagePath": img.name, "imageData": None,
             "imageHeight": 240, "imageWidth": 320}
        if flags is None:
            d.pop("flags")
        (root / f"{name}.json").write_text(json.dumps(d))
    return root


MESSY = [("a0", {"cat": True, "dog": False}, []), ("a1", {"cat": False, "dog": True}, []),
         ("a2", {"cat": True, "dog": False}, []), ("a3", {"cat": True, "dog": True}, []),
         ("a4", {"cat": False, "dog": False}, []), ("a5", {"cat": 1, "dog": 0}, []), ("a6", {"cat": "true"}, []),
         ("a7", {"Cat": True}, []), ("a8", None, []), ("a9", {"bird": True}, [])]
BOX = [{"label": "car", "points": [[10, 10], [60, 60]], "shape_type": "rectangle", "group_id": None,
        "description": "", "flags": {}}]


def test_detection(tmp_path):
    from AIO_QC.core.dataset import DatasetType as D, detect_dataset_types, open_dataset
    r = tmp_path / "tt"
    for sp in ("train", "test"):
        FX.classification(r / sp, n=2)
    assert detect_dataset_types(r) == [D.CLS_FOLDERS]
    a = open_dataset(r, D.CLS_FOLDERS)
    smp = a.discover()
    assert a.class_names == ["cat", "dog"] and {x.split for x in smp} == {"train", "test"}
    # a stray notes.txt / a runs folder / one JSON do not hide class folders
    r2 = Path(FX.classification(tmp_path / "stray", n=4))
    (r2 / "cat" / "notes.txt").write_text("x")
    (r2 / "meta.json").write_text("{}")
    (r2 / "runs" / "train").mkdir(parents=True)
    FX._img(str(r2 / "runs" / "train" / "results.png"), seed=3)
    t = detect_dataset_types(r2)
    assert t[0] == D.CLS_FOLDERS
    a = open_dataset(r2, D.CLS_FOLDERS)
    a.discover()
    assert a.class_names == ["cat", "dog"]
    # "train" next to "cat": train is a class
    r3 = Path(FX.classification(tmp_path / "trainclass", n=2))
    (r3 / "train").mkdir()
    FX._img(str(r3 / "train" / "x.jpg"), seed=9)
    a = open_dataset(r3, D.CLS_FOLDERS)
    a.discover()
    assert a.class_names == ["cat", "dog", "train"]
    # flags: a messy set is still offered as classification; boxes + flags offered as classification too
    assert detect_dataset_types(lm_flags(tmp_path / "messy", MESSY))[0] == D.CLS_LABELME
    mixed = detect_dataset_types(lm_flags(tmp_path / "mixed", [(f"m{i}", {"day": True}, BOX) for i in range(4)]))
    assert mixed == [D.LABELME, D.CLS_LABELME]


def test_flag_values_and_mixed(tmp_path):
    from AIO_QC.core.dataset import DatasetType as D, open_dataset
    a = open_dataset(lm_flags(tmp_path / "f", MESSY), D.CLS_LABELME)
    got = {}
    for x in a.discover():
        ann = a.load(x)
        got[x.name] = (ann.tags, [k for k, *_r in ann.flag_issues])
    assert got["a5.json"] == (["cat"], ["value", "value"]) and got["a6.json"] == (["cat"], ["value"])
    assert got["a8.json"] == ([], ["no_flags"]) and got["a4.json"] == ([], [])
    # saved: proper true / false
    x = next(s for s in a.discover() if s.name == "a5.json")
    a.set_tags(x, ["dog"])
    assert json.loads(Path(x.ann_path).read_text())["flags"] == {"cat": False, "dog": True}
    m = open_dataset(lm_flags(tmp_path / "m", [("m0", {"day": True}, BOX)]), D.CLS_LABELME)
    ann = m.load(m.discover()[0])
    assert ann.label_counts() == {"day": 1} and len(ann.shapes) == 1           # boxes kept, not classes
    m.set_tags(m.discover()[0], ["night"])
    d = json.loads((tmp_path / "m" / "m0.json").read_text())
    assert d["flags"] == {"day": False, "night": True} and d["shapes"][0]["label"] == "car"


def _open_as(sh, qtbot, root, dtype):
    w = sh.window("image")
    w.open_dataset(Path(root), dtype)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    return w


def test_flags_change_class_undo_keys(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.widgets import class_picker as CP
    root = lm_flags(tmp_path / "f", MESSY)
    w = _open_as(shell, qtbot, root, D.CLS_LABELME)
    s = w.session
    qtbot.waitUntil(lambda: bool(s.cls_settings().get("mode")), timeout=5000)
    assert s.cls_mode == "single"                     # asked (answered "one class per image" by the harness)
    assert s.cls_classes() == ["bird", "Cat", "cat", "dog"]
    k0 = str(root / "a0.json")
    w.file_dock.select_key(k0)
    qtbot.waitUntil(lambda: w.current_key == k0, timeout=5000)
    # 1 … 9: that class (classes in order: bird, Cat, cat, dog)
    names = s.cls_classes()
    w._relabel_key(names.index("dog") + 1)
    assert s.tags_of(k0) == ["dog"] and json.loads(Path(k0).read_text())["flags"]["dog"] is True
    assert w.a.undo_edit.isEnabled() and "dog" in w.a.undo_edit.text()
    w.inline.undo()
    assert s.tags_of(k0) == ["cat"] and json.loads(Path(k0).read_text())["flags"] == {"cat": True, "dog": False}
    w.inline.redo()
    assert s.tags_of(k0) == ["dog"]
    # the picker: one class
    monkeypatch.setattr(CP.ClassPicker, "exec", lambda self: (setattr(self, "result", ("set", ["bird"])), 1)[1])
    w.change_class()
    assert s.tags_of(k0) == ["bird"]
    # multi-label: the picker adds / removes; 1 … 9 toggle
    s.save_cls_settings(mode="multi")
    monkeypatch.setattr(CP.ClassPicker, "exec",
                        lambda self: (setattr(self, "result", ("change", ["cat"], ["bird"])), 1)[1])
    w.change_class()
    assert s.tags_of(k0) == ["cat"]
    w.class_key(names.index("dog") + 1)
    assert s.tags_of(k0) == ["cat", "dog"]
    w.class_key(names.index("dog") + 1)
    assert s.tags_of(k0) == ["cat"]
    from AIO_QC.core.backups import BackupStore
    assert BackupStore(root).exists()


def test_picker_widget(qtbot):
    from PyQt6.QtCore import Qt
    from AIO_QC.widgets.class_picker import ClassPicker
    p = ClassPicker(None, ["cat", "dog", "bird"], {"cat": 2}, n=2)
    qtbot.addWidget(p)
    p.search.setText("bi")
    p._enter()
    assert p.result == ("set", ["bird"])
    p = ClassPicker(None, ["cat", "dog"], {"cat": 2}, n=2)
    qtbot.addWidget(p)
    p.search.setText("horse")
    assert "new class" in p.new_hint.text()
    p._enter()
    assert p.result == ("set", ["horse"])
    m = ClassPicker(None, ["cat", "dog", "bird"], {"cat": 2, "dog": 1}, n=2, multi=True)
    qtbot.addWidget(m)
    assert m.list.item(1).checkState() == Qt.CheckState.PartiallyChecked
    m._toggle(m.list.item(0))                       # cat off
    m._toggle(m.list.item(2))                       # bird on
    m._enter()
    assert m.result == ("change", ["bird"], ["cat"])


def test_folders_change_class_undo(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.core.state import REVIEW_FLAG
    root = Path(FX.classification(tmp_path / "c", n=3))
    w = _open_as(shell, qtbot, root, D.CLS_FOLDERS)
    s = w.session
    k = str(root / "cat" / "cat_1.jpg")
    s.set_review([k], REVIEW_FLAG, "check")
    done, errors = s.set_class([k], "dog")
    new = str(root / "dog" / "cat_1.jpg")
    assert done == [new] and not errors and Path(new).is_file() and s.review.get(new)["note"] == "check"
    assert s.set_tags([new], tags=["a", "b"])[1]          # a class folder image has exactly one class
    s.undo_class()
    assert Path(k).is_file() and not Path(new).exists() and k in s.by_key and s.review.get(k)["note"] == "check"
    s.redo_class()
    assert Path(new).is_file() and new in s.by_key
    s.undo_class()
    assert os.path.isfile(k)
    shutil.rmtree(root / "dog")


def test_classification_check(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.widgets import classification_check_dialog as CC
    root = lm_flags(tmp_path / "f", MESSY)
    w = _open_as(shell, qtbot, root, D.CLS_LABELME)
    s = w.session
    qtbot.waitUntil(lambda: bool(s.cls_settings().get("mode")), timeout=5000)
    w.open_tool("cls_check")
    d = w.open_dialogs["cls_check"]
    qtbot.waitUntil(lambda: d.worker is None and bool(d.rows), timeout=30000)
    by = {}
    for r in d.rows:
        by.setdefault(r.cells[0], []).append(r)
    reasons = {r.key.rsplit("/", 1)[-1]: r.cells[3] for r in by[CC.NO_CLASS]}
    assert reasons == {"a4.json": "every flag is false", "a8.json": "the file has no flags"}
    assert [r.key.rsplit("/", 1)[-1] for r in by[CC.SEVERAL]] == ["a3.json"]
    assert {r.key.rsplit("/", 1)[-1] for r in by[CC.VALUES]} == {"a5.json", "a6.json"}
    assert [r.cells[1] for r in by[CC.VARIANT]] == ["Cat"]
    assert {r.cells[1] for r in by[CC.RARE]} >= {"bird", "Cat"}
    # fix the values: written as true / false, one step of Undo
    for r in by[CC.VALUES]:
        d.mark(r) if hasattr(d, "mark") else None
    d.fix_values()
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    assert json.loads((root / "a5.json").read_text())["flags"] == {"cat": True, "dog": False}
    assert json.loads((root / "a6.json").read_text())["flags"] == {"cat": True}
    assert not any(r.cells[0] == CC.VALUES for r in d.rows if r)
    s.undo_class()
    assert json.loads((root / "a5.json").read_text())["flags"] == {"cat": 1, "dog": 0}
    # merge Cat into cat
    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getItem", staticmethod(lambda *a, **k: ("cat", True)))
    d.merge()
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    assert json.loads((root / "a7.json").read_text())["flags"] == {"Cat": False, "cat": True}
    # multi-label: several classes are fine
    d.mode.setCurrentIndex(d.mode.findData("multi"))
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    assert s.cls_mode == "multi" and not any(r.cells[0] == CC.SEVERAL for r in d.rows if r)
    d.close()


def test_classification_check_folders(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.widgets import classification_check_dialog as CC
    r = tmp_path / "s"
    FX.classification(r / "train", n=3)
    FX.classification(r / "val", n=1)
    shutil.rmtree(r / "val" / "dog")
    (r / "train" / "Dog").mkdir()
    FX._img(str(r / "train" / "Dog" / "x.jpg"), seed=33)
    w = _open_as(shell, qtbot, r, D.CLS_FOLDERS)
    w.open_tool("cls_check")
    d = w.open_dialogs["cls_check"]
    qtbot.waitUntil(lambda: d.worker is None and bool(d.rows), timeout=30000)
    kinds = {(x.cells[0], x.cells[1]) for x in d.rows if x}
    assert (CC.SPLIT, "dog") in kinds and (CC.VARIANT, "Dog") in kinds
    assert not d.mode.isEnabled()
    d.close()


def test_classes_dialog_flags(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.widgets.cls_classes_dialog import ClsClassesDialog
    root = lm_flags(tmp_path / "f", MESSY)
    w = _open_as(shell, qtbot, root, D.CLS_LABELME)
    s = w.session
    w.open_tool("remove_class")
    d = w.open_dialogs["remove_class"]
    assert isinstance(d, ClsClassesDialog) and d.order == ["bird", "Cat", "cat", "dog"]
    d.table.selectRow(1)
    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getItem", staticmethod(lambda *a, **k: ("cat", True)))
    d.merge()                                                       # Cat → cat
    d.table.selectRow(0)
    d.remove()                                                      # bird removed
    d.table.selectRow(3)
    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getText", staticmethod(lambda *a, **k: ("canine", True)))
    d.rename()                                                      # dog → canine
    d.sort_count()
    d.apply()
    assert json.loads((root / "a7.json").read_text())["flags"] == {"cat": True}
    assert json.loads((root / "a9.json").read_text())["flags"] == {}
    assert json.loads((root / "a1.json").read_text())["flags"] == {"cat": False, "canine": True}
    assert s.cls_settings()["order"][0] == "cat" and "bird" not in s.cls_classes()
    s.undo_class()                                                  # the last step: dog → canine
    assert json.loads((root / "a1.json").read_text())["flags"] == {"cat": False, "dog": True}


def test_classes_dialog_folders(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    root = Path(FX.classification(tmp_path / "c", n=2))
    (root / "Dog").mkdir()
    FX._img(str(root / "Dog" / "z.jpg"), seed=21)
    w = _open_as(shell, qtbot, root, D.CLS_FOLDERS)
    w.open_tool("remove_class")
    d = w.open_dialogs["remove_class"]
    d.table.selectRow(d.order.index("Dog"))
    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getItem", staticmethod(lambda *a, **k: ("dog", True)))
    d.merge()
    d.table.selectRow(d.order.index("cat"))
    d.remove()
    d.apply()
    assert (root / "dog" / "z.jpg").is_file() and not (root / "Dog").exists()
    assert not (root / "cat").exists() and any((root / ".aio_qc_trash").rglob("cat_0.jpg"))
    assert w.session.cls_classes() == ["dog"]


def test_class_panel(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    """The Annotations dock of a classification dataset is the Class panel: the image's class ticked, one click moves
    it (class folders), several selected images show partial classes, New class…, Ctrl+Z and the panel follows."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QInputDialog
    from AIO_QC.core.dataset import DatasetType as D
    root = Path(FX.classification(tmp_path / "p", n=3))
    w = _open_as(shell, qtbot, root, D.CLS_FOLDERS)
    s, cp = w.session, w.class_panel
    k = str(root / "cat" / "cat_1.jpg")
    w.file_dock.select_key(k)
    qtbot.waitUntil(lambda: w.current_key == k, timeout=5000)
    assert w._ann_stack.currentWidget() is cp and w.annotations_title.title.text() == "Class"

    def item(name):
        return next(cp.list.item(r) for r in range(cp.list.count())
                    if cp.list.item(r).data(Qt.ItemDataRole.UserRole) == name)
    assert item("cat").checkState() == Qt.CheckState.Checked and item("dog").checkState() == Qt.CheckState.Unchecked
    cp._clicked(item("dog"))                                     # one click: the file moves into dog/
    new = str(root / "dog" / "cat_1.jpg")
    assert Path(new).is_file() and not Path(k).exists() and w.current_key == new
    assert item("dog").checkState() == Qt.CheckState.Checked
    s.undo_class()                                               # Ctrl+Z: back, and the panel follows
    assert Path(k).is_file()
    w.file_dock.select_key(k)
    qtbot.waitUntil(lambda: w.current_key == k, timeout=5000)
    assert item("cat").checkState() == Qt.CheckState.Checked
    # several images selected: a dash for a class only some have
    k2 = str(root / "dog" / "dog_1.jpg")
    monkeypatch.setattr(w.file_dock, "selected_keys", lambda: [k, k2])
    cp.refresh()
    assert "2 images selected" in cp.head.text()
    assert item("cat").checkState() == Qt.CheckState.PartiallyChecked
    cp.filter.setText("do")                                      # filter + Enter: the first match for both
    assert item("cat").isHidden() and not item("dog").isHidden()
    cp._take_first()
    assert s.tags_of(str(root / "dog" / "cat_1.jpg")) == ["dog"] and Path(k2).is_file()
    monkeypatch.setattr(w.file_dock, "selected_keys", lambda: [])
    # New class…: a new folder
    w.file_dock.select_key(k2)
    qtbot.waitUntil(lambda: w.current_key == k2, timeout=5000)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("horse", True))
    cp._new_class()
    assert (root / "horse" / "dog_1.jpg").is_file() and "horse" in s.cls_classes()
