# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Class Gallery: review orders (model findings, look), confusions, look-alikes, sample sign-off (core) — and the
window on a class-folders dataset: one class as tiles, 1 … 9 / drag moves, Ctrl+Z, compare, the model's badges and
class, Space accepts the screen, confusion cell, look-alikes, sample check + sign off, Crop Gallery redirect."""
from pathlib import Path

import numpy as np

from AIO_QC.core import cls_review as CR

from . import fixtures as FX
from .test_aio_image import shell  # noqa: F401


# ---------------------------------------------------------------------------------------------------- core
def test_model_view_and_orders():
    tags = {"a": ["cat"], "b": ["cat"], "c": ["cat"], "d": ["cat"], "e": ["cat"]}.get
    preds = {"a": [("cat", 0.95)], "b": [("dog", 0.91), ("cat", 0.09)], "c": [("dog", 0.6), ("cat", 0.4)],
             "d": [("cat", 0.42), ("dog", 0.3)]}
    assert CR.model_view(["cat"], preds["b"])[:3] == (3, 0.91, "dog")
    assert CR.model_view(["cat"], preds["c"])[0] == 2
    assert CR.model_view(["cat"], preds["d"])[0] == 1
    assert CR.model_view(["cat"], preds["a"])[0] == 0 and CR.model_view([], preds["a"])[0] == 3
    keys = list("abcde")
    assert CR.order_keys(keys, CR.SUSPICIOUS, tags, preds)[:3] == ["b", "c", "d"]
    assert CR.order_keys(keys, CR.LEAST_SURE, tags, preds) == ["b", "c", "d", "a", "e"]     # e: no prediction, last
    # look: e is the odd one out
    rows = {k: i for i, k in enumerate(keys)}
    vecs = np.array([[1, 0], [1, 0.05], [1, -0.05], [1, 0.02], [0, 1]], dtype=np.float32)
    assert CR.order_keys(keys, CR.LEAST_TYPICAL, tags, rows=rows, vecs=vecs)[0] == "e"
    sus = CR.order_keys(keys, CR.SUSPICIOUS, tags, {}, rows, vecs)        # no model: by look
    assert sus[0] == "e"
    assert sorted(CR.order_keys(keys, CR.BY_LOOK, tags, rows=rows, vecs=vecs)) == keys
    assert sorted(CR.order_keys(keys, CR.RANDOM, tags, seed=3)) == keys
    conf = CR.confusions(keys, tags, preds)
    assert conf == {("cat", "dog"): ["b", "c"]}


def test_lookalikes_and_signoff():
    rows = {"c1": 0, "c2": 1, "d1": 2, "d2": 3}
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0.999, 0.01, 0], [0, 0, 1]], dtype=np.float32)
    found = CR.lookalikes(["c1", "c2"], ["d1", "d2"], rows, vecs, 0.95, chunk=1)
    assert list(found) == ["c1"] and found["c1"][0] == "d1" and found["c1"][1] > 0.99
    keys = [f"k{i}" for i in range(1000)]
    smp = CR.sample_for(keys, 0.10, seed=1)
    assert 80 <= len(smp) <= 95 and len(set(smp)) == len(smp)
    e = CR.signoff(88, 2, 1000)
    assert e["est"] == round(2 / 88 * 1000) and e["est_lo"] < e["est"] < e["est_hi"] and e["hi"] < 0.1


# ---------------------------------------------------------------------------------------------------- window
def _dataset(root: Path, n=8):
    for c in ("cat", "dog", "bird"):
        (root / c).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            FX._img(str(root / c / f"{c}_{i}.jpg"), seed=i + {"cat": 0, "dog": 20, "bird": 40}[c])
    return root


def _open(sh, qtbot, root):
    from AIO_QC.core.dataset import DatasetType as D
    w = sh.window("image")
    w.open_dataset(Path(root), D.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    w.open_tool("class_gallery")
    gal = w.open_dialogs["class_gallery"]
    qtbot.waitUntil(lambda: gal._preds_worker.isFinished() and "reading" not in gal.model_lbl.text(),
                    timeout=10000)
    return w, gal


def test_gallery_moves_undo_compare(shell, qtbot, tmp_path):  # noqa: F811
    root = _dataset(tmp_path / "g")
    w, gal = _open(shell, qtbot, root)
    s = w.session
    g = gal.grids[0]
    assert gal.windowTitle() == "Class Gallery"
    assert g.cls in ("cat", "dog", "bird") and g.m.rowCount() == 8
    gal.show_class("cat")
    assert g.m.rowCount() == 8 and gal.classes.currentItem().data(Qt_user()) == "cat"
    k = str(root / "cat" / "cat_3.jpg")
    g.setCurrentIndex(g.m.index(g.m.pos[k]))
    names = s.cls_classes()
    gal.class_key(names.index("dog") + 1)                       # 1 … 9: the file moves; the tile leaves the class
    new = str(root / "dog" / "cat_3.jpg")
    qtbot.waitUntil(lambda: g.m.rowCount() == 7, timeout=5000)
    assert Path(new).is_file() and k not in g.m.pos
    assert g.currentIndex().isValid()                            # the keyboard stays in the grid
    gal.undo()                                                   # Ctrl+Z: back in the class (at the front)
    qtbot.waitUntil(lambda: g.m.rowCount() == 8, timeout=5000)
    assert Path(k).is_file() and g.m.keys[0] == k
    # compare: dog beside cat, a move shows up on the right at once
    gal.compare_box.setCurrentIndex(gal.compare_box.findData("dog"))
    g2 = gal.grids[1]
    assert not gal.right_col.isHidden() and g2.cls == "dog" and g2.m.rowCount() == 8
    gal.move_keys([k], "dog", source="cat")
    qtbot.waitUntil(lambda: g2.m.rowCount() == 9 and g.m.rowCount() == 7, timeout=5000)
    assert g2.m.keys[0] == new
    gal.undo()
    qtbot.waitUntil(lambda: g.m.rowCount() == 8 and g2.m.rowCount() == 8, timeout=5000)
    # drop data: what a drag carries
    from PyQt6.QtCore import QMimeData
    from AIO_QC.widgets.class_gallery import MIME
    md = QMimeData()
    md.setData(MIME, "\n".join([k]).encode())
    assert bytes(md.data(MIME)).decode() == k
    # the class list: counts and reviewed share
    item = next(gal.classes.item(i) for i in range(gal.classes.count()) if gal.classes.item(i).data(Qt_user()) == "cat")
    assert item.data(Qt_user() + 1) == "8"
    # Crop Gallery in a classification dataset opens this window
    w.open_tool("crop_gallery")
    assert "crop_gallery" not in w.open_dialogs and w.open_dialogs["class_gallery"] is gal
    gal.close()


def Qt_user():
    from PyQt6.QtCore import Qt
    return Qt.ItemDataRole.UserRole


def test_gallery_model_space_sample(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import REVIEW_FLAG, REVIEW_OK
    root = _dataset(tmp_path / "m")
    w, gal = _open(shell, qtbot, root)
    s = w.session
    gal.resize(1100, 760)
    gal.show()
    qtbot.waitExposed(gal)
    gal.show_class("cat")
    g = gal.grids[0]
    bad = str(root / "cat" / "cat_5.jpg")
    unsure = str(root / "cat" / "cat_2.jpg")
    preds = {k: [("cat", 0.97), ("dog", 0.03)] for k in g.m.keys}
    preds[bad] = [("dog", 0.93), ("cat", 0.07)]
    preds[unsure] = [("cat", 0.41), ("bird", 0.35)]
    gal.sort_box.setCurrentIndex(gal.sort_box.findText(CR.SUSPICIOUS))
    gal._on_preds(("my-cls", preds))
    assert g.m.keys[:2] == [bad, unsure]                         # most suspicious first
    assert gal.model_badge(bad) == ("→ dog 93%", "error") and gal.model_badge(unsure)[0] == "? 41%"
    assert "my-cls" in gal.model_lbl.text()
    # the confusion cell as an image list
    gal.show_confusion("cat", "dog")
    assert g.m.keys == [bad] and gal.view["kind"] == "list" and not gal.back_btn.isHidden()
    g.setCurrentIndex(g.m.index(0))
    gal.use_model()                                               # M: the model's class
    new = str(root / "dog" / "cat_5.jpg")
    qtbot.waitUntil(lambda: Path(new).is_file(), timeout=5000)
    qtbot.waitUntil(lambda: g.m.keys == [new], timeout=5000)      # a list view keeps the (renamed) tile
    assert new in gal.preds                                        # its prediction followed it
    gal.undo()
    qtbot.waitUntil(lambda: Path(bad).is_file(), timeout=5000)
    gal._back()
    assert gal.view == {"kind": "class", "cls": "cat"}
    # Space: everything on screen without a mark becomes OK
    qtbot.wait(50)
    vis = gal.visible_keys(g)
    assert vis
    s.set_review([vis[0]], REVIEW_FLAG, "wrong")
    gal.accept_visible()
    assert s.review.status(vis[0]) == REVIEW_FLAG and all(s.review.status(k) == REVIEW_OK for k in vis[1:])
    # Space at the end of the list goes back to images that were scrolled past
    s.set_review([x.key for x in s.samples], None)
    g.verticalScrollBar().setValue(g.verticalScrollBar().maximum())
    for _ in range(6):
        gal.accept_visible()
    assert all(s.review.status(k) is not None for k in g.m.keys)
    # sample check of dog and sign off
    s.set_review([x.key for x in s.samples], None)
    gal.start_sample("dog", margin=0.30)
    keys = list(g.m.keys)
    assert gal.sample is not None and keys and set(keys) <= set(gal.members("dog"))
    assert not gal.sign_btn.isEnabled()
    s.set_review(keys[:1], REVIEW_FLAG, "not a dog")
    s.set_review(keys[1:], REVIEW_OK)
    gal._update_status()
    assert gal.sign_btn.isEnabled() and "1 wrong" in gal.status.text()
    gal.sign_off()
    so = s.cls_settings()["signoff"]["dog"]
    assert so["wrong"] == 1 and so["checked"] == len(keys) and so["size"] == 8
    assert gal.sample is None and gal.view["cls"] == "dog"
    dog = next(gal.classes.item(i) for i in range(gal.classes.count())
               if gal.classes.item(i).data(Qt_user()) == "dog")
    assert dog.data(Qt_user() + 3) and "signed off" in dog.toolTip()
    gal.close()


def test_gallery_lookalikes_and_vectors_follow(shell, qtbot, tmp_path):  # noqa: F811
    root = _dataset(tmp_path / "l", n=4)
    w, gal = _open(shell, qtbot, root)
    s = w.session
    keys = [x.key for x in s.samples]
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(keys), 16)).astype(np.float32)
    c0, d0 = str(root / "cat" / "cat_0.jpg"), str(root / "dog" / "dog_0.jpg")
    vecs[keys.index(c0)] = vecs[keys.index(d0)] + 0.001         # cat_0 is (nearly) the same image as dog_0
    w.embed_index = {"model": "test", "map": {(k, -1): i for i, k in enumerate(keys)}, "vecs": vecs}
    gal.show_lookalikes("cat", 0.95)
    assert gal.grids[0].m.keys == [c0] and gal.lookalike_of[c0][0] == d0
    import AIO_QC.widgets.class_gallery as CGm                    # a big class: the same, off the window's thread
    gal.lookalike_of.clear()
    real = CGm.CR.lookalikes
    try:
        gal.show_class("dog")
        orig_members = gal.members
        gal.members = lambda c: orig_members(c) * 600_000 if c == "cat" else orig_members(c)
        CGm.CR.lookalikes = lambda mine, *a, **k: real(list(dict.fromkeys(mine)), *a, **k)
        gal.show_lookalikes("cat", 0.95)
        assert not gal.look_btn.isEnabled()
        qtbot.waitUntil(lambda: gal.look_btn.isEnabled(), timeout=10000)
        assert gal.grids[0].m.keys == [c0]
    finally:
        CGm.CR.lookalikes = real
        del gal.members
    assert "≈" in gal.tooltip(c0)
    # least typical uses the vectors
    gal.sort_box.setCurrentIndex(gal.sort_box.findText(CR.LEAST_TYPICAL))
    gal.show_class("dog")
    assert gal.grids[0].m.rowCount() == 4
    # a class change moves the image: its vector follows (look sorts keep working)
    s.set_class([c0], "bird")
    new = str(root / "bird" / "cat_0.jpg")
    assert (new, -1) in w.embed_index["map"] and (c0, -1) not in w.embed_index["map"]
    gal.close()


def test_gallery_peek_confusion_cell_and_tidy_ui(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    root = _dataset(tmp_path / "p", n=4)
    w, gal = _open(shell, qtbot, root)
    gal.resize(1000, 700)
    gal.show()
    qtbot.waitExposed(gal)
    gal.show_class("cat")
    g = gal.grids[0]
    g.setCurrentIndex(g.m.index(0))
    gal.peek_press()                                              # P: the image at full size …
    assert gal._peek is not None and gal._peek.isVisible() and not gal._peek.pixmap().isNull()
    gal._peek_t -= 1.0
    gal.peek_release()                                            # … held: it closes on release
    assert not gal._peek.isVisible()
    # the confusion matrix of Model Check: double-click a cell -> those images here
    from AIO_QC.widgets.cls_model_check_dialog import ConfusionDialog
    k = str(root / "cat" / "cat_1.jpg")
    gal._on_preds(("m", {k: [("dog", 0.9), ("cat", 0.1)]}))
    opened = []
    dlg = ConfusionDialog(gal, {"cat": {"dog": 1, "cat": 3}}, open_cell=lambda a, b: opened.append((a, b)) or
                          gal.show_confusion(a, b))
    classes = [dlg.table.horizontalHeaderItem(i).text() for i in range(dlg.table.columnCount())]
    dlg.table.cellDoubleClicked.emit(classes.index("cat"), classes.index("dog"))
    assert opened == [("cat", "dog")] and g.m.keys == [k] and gal.compare_box.currentData() == "dog"
    assert not gal.right_col.isHidden() and gal.grids[1].cls == "dog"      # the model's class beside the list
    # the main window for a classifier: the Shapes tab is "Images" (classifier versions), the class on the filmstrip
    shapes = next(i for i in range(w.tabs.count()) if w.tabs.bar.tabText(i) == "Images")
    assert w.tabs.bar.isTabVisible(shapes)
    assert w.qc_actions["class_gallery"].isEnabled() and "Ctrl+Shift+G" in w.qc_actions["class_gallery"].toolTip()
    assert callable(w.filmstrip.class_colors)
    gal.close()


def test_shapes_tab_back_for_detection(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType as D
    root = _dataset(tmp_path / "c", n=2)
    w = shell.window("image")
    w.open_dataset(Path(root), D.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    shapes = next(i for i in range(w.tabs.count()) if w.tabs.bar.tabText(i) == "Images")
    assert w.qc_actions["rules"].property("aio_type_hidden")
    lm = FX.labelme(str(tmp_path / "lm"), n=3)
    w.open_dataset(Path(lm))
    qtbot.waitUntil(lambda: w.session is not None and not w.session.is_classification, timeout=30000)
    assert w.tabs.bar.tabText(shapes) == "Shapes" and not w.qc_actions["rules"].property("aio_type_hidden")
    assert w.tabs.bar.isTabVisible(shapes) and w.qc_actions["crop_gallery"].isEnabled()
    assert not w.qc_actions["class_gallery"].isEnabled()
