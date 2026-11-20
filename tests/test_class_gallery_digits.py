# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The Class Gallery on a real dataset: scikit-learn's handwritten digits (1,809 images, 10 classes) as class folders
and as LabelMe flags, with 53 planted mislabels and 12 cross-class copies. The classifier's results reach the gallery
the way Model Check leaves them (the prediction cache of a classify model in Models & environment); vectors the way
Visual Similarity leaves them. Checks that the planted errors come first, the model's sure fixes are right, copies are
found, a Space pass covers a class, a sample's estimate brackets the truth, and flags are rewritten / undone."""
import json
import shutil
from pathlib import Path

import pytest

from .test_aio_image import shell  # noqa: F401

pytest.importorskip("sklearn")
pytest.importorskip("PIL")


def _setup(sh, qtbot, tmp_path, flags=False):
    from AIO_QC.core.dataset import DatasetType as D
    from AIO_QC.core.model_check import model_signature
    from AIO_QC.core.state import save_models_config
    from AIO_QC.widgets.model_setup_dialog import server_specs
    from .digits_dataset import build
    root = tmp_path / "digits"
    meta = build(root)
    move = {p: p for p in meta["paths"]}
    if flags:                                                     # the same images as LabelMe flags
        for p in meta["paths"]:
            src = Path(p)
            dst = src.parent.parent / src.name
            shutil.move(str(src), dst)
            dst.with_suffix(".json").write_text(json.dumps(
                {"version": "5.5.0", "flags": {n: n == src.parent.name for n in meta["names"]}, "shapes": [],
                 "imagePath": dst.name, "imageData": None, "imageHeight": 96, "imageWidth": 96}))
            move[p] = str(dst)
    model = tmp_path / "digits-cls.pt"
    model.write_bytes(b"stand-in: its predictions are already in the cache")
    m = {"id": "m1", "name": "digits-cls", "path": str(model), "task": "classify", "enabled": True,
         "names": {str(i): n for i, n in enumerate(meta["names"])},
         "class_map": {str(i): n for i, n in enumerate(meta["names"])}}
    save_models_config(root, {"models": [m]})
    w = sh.window("image")
    w.open_dataset(root, D.CLS_LABELME if flags else D.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=60000)
    s = w.session
    sig = model_signature(server_specs([m], w.inference.opts, w.inference.info)[0]) + "|cls"
    for p, pr in zip(meta["paths"], meta["proba"]):
        top = sorted(enumerate(pr), key=lambda t: -t[1])[:5]
        s.cache.put_preds(sig, move[p], {"w": 96, "h": 96, "boxes": [], "probs": [[i, c] for i, c in top]})
    row = {Path(p).stem: i for i, p in enumerate(meta["paths"])}                # a sample is found by its stem
    w.embed_index = {"model": "pixel-PCA", "map": {(x.key, -1): row[Path(x.key).stem] for x in s.samples},
                     "vecs": meta["vecs"]}
    w.open_tool("crop_gallery")                                   # the Crop Gallery key opens the Class Gallery
    gal = w.open_dialogs["class_gallery"]
    gal.resize(1400, 900)
    gal.show()
    qtbot.waitExposed(gal)
    qtbot.waitUntil(lambda: gal._preds_worker.isFinished() and "checked" in gal.model_lbl.text(), timeout=30000)
    truth = _ByStem({Path(p).stem: v for p, v in meta["truth"].items()})
    return w, s, gal, truth, {Path(p).stem for p in meta["wrong"]}, {Path(p).stem for p in meta["dups"]}


class _ByStem(dict):
    """truth[key]: by the file's stem — keys change when an image moves to another class folder."""

    def __getitem__(self, k):
        return super().__getitem__(Path(k).stem)


def _stems(keys):
    return {Path(k).stem for k in keys}


def test_digits_class_folders(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core import cls_review as CR
    from AIO_QC.core.state import REVIEW_FLAG, REVIEW_OK
    w, s, gal, truth, wrong, dups = _setup(shell, qtbot, tmp_path)
    assert "1,809 image(s) checked" in gal.model_lbl.text()
    # most suspicious first: the planted mislabels are at the top of their class
    gal.sort_box.setCurrentIndex(gal.sort_box.findText(CR.SUSPICIOUS))
    pos = []
    for c in s.cls_classes():
        gal.show_class(c)
        pos += [i for i, k in enumerate(gal.grids[0].m.keys) if Path(k).stem in wrong]
    assert len(pos) == len(wrong) == 53
    assert sum(1 for i in pos if i < 10) >= 45 and sum(1 for i in pos if i < 25) >= 50
    # the confusions menu = the model's mix-ups
    gal._fill_confusions()
    assert "→" in gal.conf_menu.actions()[0].text()
    # look-alikes: every planted cross-class copy
    found = set()
    for c in s.cls_classes():
        gal.show_lookalikes(c, 0.995, wait=True)
        found |= _stems(gal.grids[0].m.keys) | _stems(v[0] for v in gal.lookalike_of.values())
    assert dups <= found
    # M on every tile where the model is sure of another class: right every time
    right = bad = 0
    for c in list(s.cls_classes()):
        gal.show_class(c)
        g = gal.grids[0]
        sure = [k for k in g.m.keys if CR.model_view(s.tags_of(k), gal.preds.get(k))[0] == 3]
        if not sure:
            continue
        g.clearSelection()
        for k in sure:
            g.selectionModel().select(g.m.index(g.m.pos[k]), g.selectionModel().SelectionFlag.Select)
        new = {k: gal.preds[k][0][0] for k in sure}
        gal.use_model()
        qtbot.waitUntil(lambda: not any(k in g.m.pos for k in sure), timeout=10000)
        for k, cls in new.items():
            assert Path(Path(k).parent.parent / cls / Path(k).name).is_file()
            right += truth[k]["true"] == cls
            bad += truth[k]["true"] != cls
    assert right >= 40 and bad == 0
    n = len(gal.members(c))
    gal.undo()
    qtbot.waitUntil(lambda: len(gal.members(c)) != n, timeout=5000)
    gal.redo()
    qtbot.waitUntil(lambda: len(gal.members(c)) == n, timeout=5000)
    # a Space pass over a class: flag what is not an eight, Space for the rest — every image reviewed
    gal.show_class("eight")
    g = gal.grids[0]
    qtbot.wait(50)
    screens = 0
    while any(s.review.status(k) is None for k in g.m.keys) and screens < 15:
        bad_here = [k for k in gal.visible_keys() if truth[k]["true"] != "eight" and s.review.status(k) is None]
        if bad_here:
            s.set_review(bad_here, REVIEW_FLAG, "not an eight")
        gal.accept_visible()
        screens += 1
    assert all(s.review.status(k) is not None for k in g.m.keys) and screens <= 10
    # sample check of 'three': the estimate brackets what is really wrong in the class
    gal.start_sample("three", margin=0.10)
    keys = sorted(gal.sample["keys"])
    off = [k for k in keys if truth[k]["true"] != "three"]
    if off:
        s.set_review(off, REVIEW_FLAG, "not a three")
    s.set_review([k for k in keys if k not in off], REVIEW_OK)
    actual = sum(1 for k in gal.members("three") if truth[k]["true"] != "three")
    gal.sign_off()
    so = s.cls_settings()["signoff"]["three"]
    assert so["est_lo"] <= actual <= so["est_hi"] and so["checked"] == len(keys)
    gal.close()


def test_digits_labelme_flags(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core import cls_review as CR
    w, s, gal, truth, wrong, dups = _setup(shell, qtbot, tmp_path, flags=True)
    gal.sort_box.setCurrentIndex(gal.sort_box.findText(CR.SUSPICIOUS))
    gal.show_class(s.cls_classes()[0])
    g = gal.grids[0]
    k = g.m.keys[0]
    tier, _w, top, _c, _o = CR.model_view(s.tags_of(k), gal.preds.get(k))
    assert tier == 3 and Path(k).stem in wrong | dups
    g.setCurrentIndex(g.m.index(0))
    gal.class_key(s.cls_classes().index(top) + 1)
    qtbot.waitUntil(lambda: k not in g.m.pos, timeout=5000)
    assert [n for n, v in json.loads(Path(k).read_text())["flags"].items() if v] == [top] == [truth[k]["true"]]
    gal.undo()
    qtbot.waitUntil(lambda: k in g.m.pos, timeout=5000)
    assert [n for n, v in json.loads(Path(k).read_text())["flags"].items() if v] == [s.cls_classes()[0]]
    gal.close()
