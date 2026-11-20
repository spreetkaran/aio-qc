# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Every feature on classification datasets (class folders, LabelMe flags; one class or several per image): what the
audit found missing or wrong, each pinned — Smart Filter, Rules on the Labels tab, tooltips naming the classifier
tool, Classification Check's same image in two classes, Empty Annotations' wording, the class on Review Grid tiles
and on batch screenshots, hand-off batches for class folders, Improve Model and Train Model for a classifier, and
JSON Lines classification tables."""
import json
import shutil
import time
from pathlib import Path

from PyQt6.QtGui import QColor, QImage, QPainter

from . import fixtures as FX
from .test_aio_image import shell  # noqa: F401

CLASSES = ["cat", "dog", "bird"]


def _flags_ds(root: Path, multi=True) -> Path:
    root.mkdir(parents=True)
    for c in CLASSES:
        for j in range(6):
            p = root / f"{c}_{j}.jpg"
            FX._img(str(p), seed=CLASSES.index(c) * 10 + j)
            flags = {k: k == c for k in CLASSES}
            if multi and c == "cat" and j < 2:
                flags["dog"] = True                               # two images with cat AND dog
            p.with_suffix(".json").write_text(json.dumps({"version": "5.5.0", "flags": flags, "shapes": [],
                                                          "imagePath": p.name, "imageData": None,
                                                          "imageHeight": 64, "imageWidth": 64}))
    return root


def _folders_ds(root: Path) -> Path:
    for c in CLASSES:
        (root / c).mkdir(parents=True)
        for j in range(6):
            FX._img(str(root / c / f"{c}_{j}.jpg"), seed=CLASSES.index(c) * 10 + j)
    shutil.copy(root / "cat" / "cat_1.jpg", root / "dog" / "copy_of_cat_1.jpg")   # the same picture, other class
    return root


def _open(sh, qtbot, root, dtype):
    w = sh.window("image")
    w.open_dataset(Path(root), dtype)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    return w


def _scan(d, qtbot):
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    return [r for r in d.rows if r]


def test_tools_on_flags_multi(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtCore import Qt
    from AIO_QC.core.dataset import DatasetType as D
    w = _open(shell, qtbot, _flags_ds(tmp_path / "f"), D.CLS_LABELME)
    s = w.session
    s.save_cls_settings(mode="multi")
    # Smart Filter: images with cat AND dog
    assert w.qc_actions["smart_filter"].isEnabled()
    w.open_tool("smart_filter")
    d = w.open_dialogs["smart_filter"]
    for lst in (d.include,):
        for i in range(lst.count()):
            if lst.item(i).text() in ("cat", "dog"):
                lst.item(i).setCheckState(Qt.CheckState.Checked)
    d.inc_mode.setCurrentIndex(1)                                        # all of
    rows = _scan(d, qtbot)
    assert sorted(Path(r.key).stem for r in rows) == ["cat_0", "cat_1"] and rows[0].cells[0] == 2
    assert "Classes" in [d.table.horizontalHeaderItem(i).text() for i in range(d.table.columnCount())]
    d.close()
    # Rules are on the Labels tab for a classifier (the Shapes tab is hidden)
    page = w._ribbon_pages["labels"]
    assert any(b.defaultAction() is w.rules_cls_act for b in page.buttons()) and w.rules_cls_act.isEnabled()
    shapes = next(i for i in range(w.tabs.count()) if w.tabs.bar.tabText(i) == "Images")   # Shapes → Images
    assert w.tabs.bar.isTabVisible(shapes)
    assert w.qc_actions["rules"].property("aio_type_hidden")                  # box rules hidden: Labels tab's
    assert all(w.qc_actions[k].isEnabled() for k in ("duplicates", "outliers", "sequence", "dim_area", "train_size"))
    w.rules_cls_act.trigger()
    assert "rules" in w.open_dialogs
    w.open_dialogs["rules"].close()
    # a box check opens its classifier version (and says so)
    assert w.qc_actions["ambiguity"].isEnabled() and "Classifier version" in w.qc_actions["ambiguity"].toolTip()
    # Empty Annotations speaks of classes
    w.open_tool("empty")
    e = w.open_dialogs["empty"]
    assert "without a class" in e.header_desc.text() and e.also_images.isChecked()
    e.close()


def test_rules_button_hidden_for_detection(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "lm"), None)
    page = w._ribbon_pages["labels"]
    assert not any(b.defaultAction() is w.rules_cls_act for b in page.buttons())
    assert "Classifier version" not in w.qc_actions["ambiguity"].toolTip()
    assert w.tabs.bar.tabText([i for i in range(w.tabs.count())][2]) == "Shapes"


def test_same_image_other_class_and_tile_chip(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.widgets.classification_check_dialog import TWICE
    w = _open(shell, qtbot, _folders_ds(tmp_path / "c"), D.CLS_FOLDERS)
    w.open_tool("cls_check")
    d = w.open_dialogs["cls_check"]
    qtbot.waitUntil(lambda: d.worker is None and any(d.rows), timeout=30000)        # its automatic scan
    twice = [r for r in d.rows if r and r.cells[0] == TWICE]
    assert sorted(Path(r.key).name for r in twice) == ["cat_1.jpg", "copy_of_cat_1.jpg"]
    assert "copy_of_cat_1.jpg (dog)" in next(r for r in twice if r.key.endswith("cat_1.jpg")
                                            and "copy" not in r.key).cells[3]
    d.close()
    # the class on a Review Grid tile (and on batch screenshots: same painter)
    from AIO_QC.widgets.review_grid import _class_chip
    img = QImage(120, 90, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("white"))
    p = QPainter(img)
    _class_chip(p, img, ["dog"], {"dog": "#ff0000"})
    p.end()
    assert QColor(img.pixel(12, 80)).lightness() < 120                   # the dark pill, bottom-left
    from AIO_QC.widgets.batch_shots import BatchShots
    k = str(tmp_path / "c" / "dog" / "dog_2.jpg")
    title, meta, _note = BatchShots._caption_text(type("X", (), {"app": w})(), k)
    assert meta.startswith("class: dog")


def test_handoff_class_folders(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core import handoff
    from AIO_QC.core.dataset import DatasetType as D
    root = _folders_ds(tmp_path / "h")
    w = _open(shell, qtbot, root, D.CLS_FOLDERS)
    s = w.session
    assert w.qc_actions["handoff_export"].isEnabled() and w.qc_actions["handoff_import"].isEnabled()
    keys = [str(root / "cat" / f"cat_{j}.jpg") for j in (2, 3, 4)]
    items = [handoff.BatchItem(k, Path(k).name, None, k, "check the class", "", [], [], tags=s.tags_of(k))
             for k in keys]
    out = tmp_path / "batch"
    st = handoff.export_batch(root, items, out, handoff.CLS_FOLDERS, title="cats")
    assert st["items"] == 3 and (out / "cat" / "cat_2.jpg").is_file() and "MOVE" in (out / "README.txt").read_text()
    # the annotator: cat_2 is a dog, cat_3 a new class; cat_4 stays
    (out / "dog").mkdir()
    shutil.move(out / "cat" / "cat_2.jpg", out / "dog" / "cat_2.jpg")
    (out / "fox").mkdir()
    shutil.move(out / "cat" / "cat_3.jpg", out / "fox" / "cat_3.jpg")
    m = handoff.load_manifest(out)
    rows = handoff.compare(root, m, out)
    by = {Path(r.entry["image"]).name: r for r in rows}
    assert by["cat_2.jpg"].state == "ready" and by["cat_2.jpg"].new_class == "dog"
    assert by["cat_3.jpg"].new_class == "fox" and by["cat_4.jpg"].state == "unchanged"
    # merge through the import window: files move (one undo step per class)
    w.open_tool("handoff_import")
    dlg = w.open_dialogs["handoff_import"]
    dlg.batch_dir, dlg.manifest, dlg.rows = out, m, rows
    monkeypatch.setattr(dlg, "_ticked", lambda: [r for r in rows if r.state == "ready"])
    monkeypatch.setattr(dlg, "_load", lambda: None)
    dlg._merge()
    assert (root / "dog" / "cat_2.jpg").is_file() and (root / "fox" / "cat_3.jpg").is_file()
    assert (root / "cat" / "cat_4.jpg").is_file()
    w.inline.undo()
    assert (root / "cat" / "cat_3.jpg").is_file()
    dlg.close()


def test_improve_and_train_for_a_classifier(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core import improve as IM
    from AIO_QC.core import training as T
    m = {"n": 40, "top1": 0.85, "top5": 1.0,
         "classes": {"cat": {"tp": 18, "fp": 1, "fn": 2, "support": 20, "precision": 0.95, "recall": 0.9, "f1": 0.92},
                     "dog": {"tp": 10, "fp": 5, "fn": 6, "support": 16, "precision": 0.67, "recall": 0.62, "f1": 0.64},
                     "fox": {"tp": 3, "fp": 0, "fn": 1, "support": 4, "precision": 1.0, "recall": 0.75, "f1": 0.86}},
         "confusion": {"dog": {"dog": 10, "cat": 6}, "cat": {"cat": 18, "dog": 2}, "fox": {"fox": 3, "dog": 1}}}
    spots = IM.weak_spots_cls(m)
    assert spots["weak"] == ["dog"] and spots["few"] == ["fox"]
    dog = next(c for c in spots["classes"] if c["label"] == "dog")
    assert dog["confused_with"] == "cat" and "Confusions (dog → cat)" in dog["advice"]
    assert "top-1 accuracy 0.85" in IM.summary_text(spots)
    recs = [{"key": "a", "gt": ["dog"], "pred": [("cat", 0.9)]}, {"key": "b", "gt": ["dog"], "pred": [("cat", 0.6)]},
            {"key": "c", "gt": ["cat"], "pred": [("cat", 0.9)]}]
    assert IM.hardest_cls(recs) == [("a", 1), ("b", 1)]
    st = {}
    IM.record_check(st, "r1", "m1.pt", spots, 40)
    spots2 = IM.weak_spots_cls(dict(m, top1=0.9))
    IM.record_check(st, "r2", "m2.pt", spots2, 40)
    assert IM.trend(st).startswith("top-1 0.85 → 0.90")
    # Train Model: a classifier's run shows accuracy, not mAP
    run = tmp_path / "runs" / "cls1"
    run.mkdir(parents=True)
    (run / "results.csv").write_text("epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss\n"
                                     "1,10,1.2,0.61,0.95,1.1\n2,20,0.8,0.78,0.99,0.9\n3,30,0.6,0.74,1.0,0.95\n")
    rows = T.epochs(run)
    assert T.best_scores(rows) == (0.78, 0.99) and T.is_cls_rows(rows)
    r = next(x for x in T.list_runs(tmp_path / "runs") if x["name"] == "cls1")
    assert r["metric"] == "top-1" and r["best"] == 0.78
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"), None)
    w.open_training(run=run)
    td = w.train_dialog
    td._draw(rows, {})
    assert td.t_map.cap.text().startswith("top-1 accuracy") and td.t_map.value.text() == "0.780"
    assert td.t_map50.cap.text() == "top-5 accuracy" and td.t_p.isHidden()
    assert td.cls_table.horizontalHeaderItem(1).text() == "Images" and td.cls_table.isColumnHidden(5)
    td.close()


def test_jsonl_classification_table(tmp_path):
    from AIO_QC.core.prep import cls_csv as CC
    from AIO_QC.core.prep.importers import detect
    p = tmp_path / "metadata.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in [{"file_name": "a.jpg", "label": "cat"},
                                                   {"file_name": "b.jpg", "label": "dog"},
                                                   {"file_name": "c.jpg", "label": ["cat", "dog"]}]))
    assert detect(p) == "csv"
    recs, order = CC.read_table(p)
    assert [(r.file if hasattr(r, "file") else r.ref, r.tags) for r in recs] == \
        [("a.jpg", ["cat"]), ("b.jpg", ["dog"]), ("c.jpg", ["cat", "dog"])] and order == ["cat", "dog"]
    assert time.time() > 0
