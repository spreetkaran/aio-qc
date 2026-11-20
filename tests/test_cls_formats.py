# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Classification formats: tables (one column, several classes in one cell, one 1 / 0 column per class, "path
label" lists with numbers + a class list), CVAT tags, Label Studio choices — imported as class folders or LabelMe
flags; exported as class folders, LabelMe flags, a CSV table, crops of every shape, and a LabelMe dataset's image
flags as class folders."""
import json
from pathlib import Path

from . import fixtures as FX


def _images(root: Path, names):
    root.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (root / n).parent.mkdir(parents=True, exist_ok=True)
        FX._img(str(root / n), seed=i)
    return root


def test_sniff_layouts(tmp_path):
    from AIO_QC.core.prep import cls_csv as CC
    a = tmp_path / "a.csv"
    a.write_text("file_name,label\nx.jpg,cat\ny.jpg,dog\n")
    s, head, _rows, desc = CC.sniff(a)
    assert (s.file_col, s.mode, s.label_col, s.sep) == (0, "column", 1, "none") and "file_name" in desc
    b = tmp_path / "b.csv"
    b.write_text("filename, cat, dog, bird\nx.jpg,1,0,1\ny.jpg,0,1,0\n")
    s, head, _r, desc = CC.sniff(b)
    assert s.mode == "onehot" and s.onehot_cols == [1, 2, 3] and "3 classes" in desc
    recs, order = CC.read_table(b, s)
    assert order == ["cat", "dog", "bird"] and recs[0].tags == ["cat", "bird"]
    c = tmp_path / "c.tsv"
    c.write_text("image\ttags\tsplit\nx.jpg\tcat;dog\ttrain\ny.jpg\tdog\tval\n")
    s, *_r = CC.sniff(c)
    assert s.sep == ";" and s.split_col == 2
    recs, _o = CC.read_table(c, s)
    assert recs[0].tags == ["cat", "dog"] and recs[1].split == "val"
    d = tmp_path / "train.txt"
    d.write_text("n01/x.jpg 0\nn02/y.jpg 1\n")
    s, head, *_r = CC.sniff(d)
    assert not s.header and s.file_col == 0
    (tmp_path / "classes.txt").write_text("tench\ngoldfish\n")
    s.names = CC.read_names(str(tmp_path / "classes.txt"))
    recs, order = CC.read_table(d, s)
    assert [r.tags for r in recs] == [["tench"], ["goldfish"]] and order == ["tench", "goldfish"]


def test_import_table_as_folders_and_flags(tmp_path):
    from AIO_QC.core.prep import common as C, importers as IM
    imgs = _images(tmp_path / "imgs", ["x.jpg", "sub/y.jpg", "z.jpg", "w.jpg"])
    t = tmp_path / "labels.csv"
    t.write_text("file,label,split\nx.jpg,cat,train\nsub/y.jpg,dog,val\nz.jpg,cat|dog,train\nmissing.jpg,cat,train\n"
                 "w.jpg,,train\n")
    P = IM.plan(t, imgs, tmp_path / "out", "auto", "cls_folders")
    rows = dict(P.rows)
    assert rows["Images found"] == 4 and rows["    not found (skipped)"] == 1 and rows["    images with several"] == 1
    st = IM.run(P)
    assert not st["errors"]
    out = tmp_path / "out"
    assert (out / "train" / "cat" / "x.jpg").is_file() and (out / "val" / "dog" / "y.jpg").is_file()
    assert (out / "train" / "cat" / "z.jpg").is_file() and (out / "train" / "dog" / "z.jpg").is_file()
    assert (out / "train" / "_unlabelled" / "w.jpg").is_file()
    assert not C.undo_run(st["undo"])["errors"] and not (out / "train" / "cat" / "x.jpg").exists()
    P = IM.plan(t, imgs, tmp_path / "out2", "csv", "cls_labelme")
    IM.run(P)
    d = json.loads((tmp_path / "out2" / "train" / "z.json").read_text())
    assert d["flags"] == {"cat": True, "dog": True} and d["shapes"] == []
    assert json.loads((tmp_path / "out2" / "train" / "x.json").read_text())["flags"] == {"cat": True, "dog": False}


def test_cvat_tags_and_label_studio_choices(tmp_path):
    from AIO_QC.core.prep import importers as IM
    imgs = _images(tmp_path / "imgs", ["a.jpg", "b.jpg"])
    x = tmp_path / "ann.xml"
    x.write_text('<annotations><image id="0" name="a.jpg" width="320" height="240"><tag label="day"/>'
                 '<box label="car" xtl="1" ytl="2" xbr="30" ybr="40"/></image>'
                 '<image id="1" name="b.jpg" width="320" height="240"><tag label="night"/></image></annotations>')
    P = IM.plan(x, imgs, tmp_path / "o1", "auto", "labelme")
    IM.run(P)
    d = json.loads((tmp_path / "o1" / "a.json").read_text())
    assert d["flags"] == {"day": True, "night": False} and d["shapes"][0]["label"] == "car"
    ls = tmp_path / "ls.json"
    ls.write_text(json.dumps([{"data": {"image": "/data/upload/1/abcd1234-a.jpg"}, "annotations": [{"result": [
        {"type": "choices", "value": {"choices": ["indoor"]}, "from_name": "c", "to_name": "image"}]}]}]))
    P = IM.plan(ls, imgs, tmp_path / "o2", "auto", "cls_folders")
    IM.run(P)
    assert (tmp_path / "o2" / "indoor" / "a.jpg").is_file()


def test_exports(tmp_path):
    from AIO_QC.core import dataset as D, export as EX
    # class folders -> LabelMe flags, and a CSV table
    root = Path(FX.classification(tmp_path / "cls", n=2))
    a = D.open_dataset(root, D.DatasetType.CLS_FOLDERS)
    smp = a.discover()
    EX.export({"train": smp}, a, root, tmp_path / "lm", "labelme", ["cat", "dog"], is_classification=True)
    j = json.loads(next((tmp_path / "lm" / "train").rglob("cat_0.json")).read_text())
    assert j["flags"] == {"cat": True, "dog": False}
    EX.export({"train": smp}, a, root, tmp_path / "tab", "cls_csv", ["cat", "dog"], is_classification=True)
    rows = (tmp_path / "tab" / "labels.csv").read_text().splitlines()
    assert rows[0] == "file,labels,split" and any(r.endswith(",cat,train") for r in rows) and len(rows) == 5
    # boxes -> crops in class folders
    lm = Path(FX.labelme(tmp_path / "det", n=3))
    b = D.open_dataset(lm, D.DatasetType.LABELME)
    smp = b.discover()
    labels = sorted({x["label"] for s in smp for x in b.load(s).shapes})
    st = EX.export({"train": smp}, b, lm, tmp_path / "crops", "crops", labels, crop={"pad": 0.1, "square": True})
    crops = list((tmp_path / "crops" / "train").rglob("*.jpg"))
    assert crops and st["links"]["crops"] == len(crops) and {c.parent.name for c in crops} <= set(labels)
    # a LabelMe dataset's image flags -> class folders
    for p in lm.glob("*.json"):
        d = json.loads(p.read_text())
        d["flags"] = {"day": p.stem.endswith("0"), "night": not p.stem.endswith("0")}
        p.write_text(json.dumps(d))
    EX.export({"train": smp}, b, lm, tmp_path / "ff", "flags_folders", [])
    assert {p.parent.name for p in (tmp_path / "ff" / "train").rglob("*.jpg")} == {"day", "night"}


def test_import_page_table_and_cls_export_dialog(qtbot, tmp_path):
    from .test_aio_image import _open
    from AIO_QC.shell import AppShell
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    sh = AppShell(app)
    app.shell = sh
    w = sh.show("image")
    qtbot.addWidget(w)
    try:
        imgs = _images(tmp_path / "imgs", ["x.jpg", "y.jpg"])
        t = tmp_path / "labels.csv"
        t.write_text("filename,cat,dog\nx.jpg,1,0\ny.jpg,0,1\n")
        w.open_data_prep()
        dp = w.open_dialogs["data_prep"]
        page = next(p for p in dp.pages if p.TITLE == "Import annotations")
        page.src.setText(str(t))
        assert not page.csv_box.isHidden() and page.c_mode.currentData() == "onehot"
        assert page.output.currentData() == "cls_folders" and "2 classes" in page.csv_desc.text()
        page.c_mode.setCurrentIndex(page.c_mode.findData("column"))
        assert page.c_label.isEnabled()
        page.c_mode.setCurrentIndex(page.c_mode.findData("onehot"))
        page.images.setText(str(imgs))
        page.out.setText(str(tmp_path / "out"))
        P = page.make_plan(None, lambda *a: None)
        page.run(P, None, lambda *a: None)
        assert (tmp_path / "out" / "cat" / "x.jpg").is_file()
        dp.close()
        # the class-folder dataset just made: every classification export from Convert & Split
        ww = _open(sh, qtbot, tmp_path / "out")
        ww.open_tool("split")
        d = ww.open_dialogs["export"]
        assert [d.fmt.itemData(i) for i in range(d.fmt.count())] == ["cls_folders", "labelme", "cls_csv"]
        for i in range(d.fmt.count()):
            d.fmt.setCurrentIndex(i)
            d.out.setText(str(tmp_path / f"exp_{d.fmt.currentData()}"))
            d.start()
            qtbot.waitUntil(lambda: d.worker is None and ("Converted" in d.status.text() or "⚠" in d.status.text()),
                            timeout=60000)
            assert "Converted" in d.status.text(), d.status.text()
        assert (tmp_path / "exp_cls_csv" / "labels.csv").is_file()
        d.close()
    finally:
        sh._closing = True
        sh.image._quit_reason = "test"
        sh.image.close()
        app.shell = None
