# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Class folders: I sets images aside in the issues folder, kept by class (and split); Ctrl+Z / Ctrl+Y; the Class
Gallery with I on one tile after another; the folder is never read as a class; put back into the class folder an
image is in now; the location from Settings; a big dataset stays quick."""
from __future__ import annotations

import csv
import os
import shutil
import time
from pathlib import Path

import pytest

from . import fixtures as FX
from .test_aio_image import shell  # noqa: F401


def _open(shell, qtbot, root):  # noqa: F811
    from AIO_QC.core.dataset import DatasetType
    w = shell.window("image")
    w.open_dataset(Path(root), DatasetType.CLS_FOLDERS)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    return w


@pytest.fixture(autouse=True)
def _default_place(qapp):
    from AIO_QC.utils import app_settings
    st = app_settings()
    old = (st.value("cls/issues_place"), st.value("cls/issues_dir"))
    st.setValue("cls/issues_place", "dataset")
    yield
    for k, v in zip(("cls/issues_place", "cls/issues_dir"), old):
        st.remove(k) if v is None else st.setValue(k, v)


def test_core_locations_names_and_return_plan(tmp_path):
    from AIO_QC.core import issues_folder as IF
    root = tmp_path / "ds"
    root.mkdir()
    assert IF.location(root) == root.resolve() / "issues"
    assert IF.location(root, "beside") == (tmp_path / "ds_issues").resolve()
    assert IF.location(root, "custom", str(tmp_path / "x")) == (tmp_path / "x").resolve() / "ds"
    assert IF.location(root, "custom", "") == root.resolve() / "issues"       # no folder chosen: the default
    f = IF.ensure(IF.location(root), root)
    assert IF.is_issues_dir(f)
    d1 = IF.destination(f, "cat", "", "a.jpg")
    d1.parent.mkdir(parents=True)
    d1.write_bytes(b"x")
    assert IF.destination(f, "cat", "", "a.jpg").name == "a_1.jpg"
    assert IF.destination(f, "", "train", "b.jpg") == f / "train" / IF.NO_CLASS / "b.jpg"
    plan = IF.return_plan(f, [d1], {"": root}, root)
    assert plan == [(d1, root / "cat" / "a.jpg", "cat", "")]


def test_send_undo_redo_and_never_a_class(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.classification(str(tmp_path / "cls"), n=6)
    w = _open(shell, qtbot, root)
    s = w.session
    assert w.a.send_issues.isEnabled() and w.a.send_issues.shortcut().toString() == "I"
    keys = sorted(k for k in s.by_key if "/cat/" in k.replace(os.sep, "/"))[:2]
    w.file_dock.select_key(keys[0])
    qtbot.waitUntil(lambda: w.current_key == keys[0], timeout=5000)
    n0 = len(s.samples)
    w.a.send_issues.trigger()                                        # I on the image shown
    issues = Path(root).resolve() / "issues"
    assert (issues / "cat" / Path(keys[0]).name).is_file() and not Path(keys[0]).exists()
    assert len(s.samples) == n0 - 1 and keys[0] not in s.by_key
    assert w.current_key is not None and w.current_key != keys[0]   # the next image opens
    assert (issues / ".aio_qc_issues").is_file()
    rows = list(csv.DictReader(open(issues / "issues_log.csv", encoding="utf-8")))
    assert rows[-1]["action"] == "to issues" and rows[-1]["class"] == "cat"
    # several at once: one step of Undo
    w.send_to_issues(keys=[keys[1], next(k for k in s.by_key if "/dog/" in k.replace(os.sep, "/"))])
    assert len(s.samples) == n0 - 3 and (issues / "dog").is_dir()
    w.inline.undo()                                                   # Ctrl+Z: both back
    assert len(s.samples) == n0 - 1 and Path(keys[1]).is_file()
    w.inline.undo()
    assert len(s.samples) == n0 and Path(keys[0]).is_file() and not any(issues.rglob("*.jpg"))
    w.inline.redo()                                                   # Ctrl+Y: aside again
    assert not Path(keys[0]).exists() and (issues / "cat" / Path(keys[0]).name).is_file()
    # the issues folder is not a class: reload the dataset, check the class list and watcher lookups
    w.reload_dataset() if hasattr(w, "reload_dataset") else w.open_dataset(Path(root), s.type)
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)
    s = w.session
    assert "issues" not in s.cls_classes() and len(s.samples) == n0 - 1
    smp, reload_ = s.base_adapter.sample_for_path(issues / "cat" / Path(keys[0]).name)
    assert smp is None and not reload_
    from AIO_QC.core.dataset import DatasetType, detect_dataset_types
    assert DatasetType.CLS_FOLDERS in detect_dataset_types(Path(root))


def test_put_back_into_the_class_it_is_in_now(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.classification(str(tmp_path / "cls"), n=4)
    w = _open(shell, qtbot, root)
    s = w.session
    cat = sorted(k for k in s.by_key if "/cat/" in k.replace(os.sep, "/"))[:2]
    w.send_to_issues(keys=cat)
    issues = Path(root).resolve() / "issues"
    (issues / "dog").mkdir()                                          # its class fixed in the issues folder
    shutil.move(str(issues / "cat" / Path(cat[0]).name), str(issues / "dog" / Path(cat[0]).name))
    w.return_issues()                                                 # (the question is answered Yes)
    assert (Path(root) / "dog" / Path(cat[0]).name).is_file() and (Path(root) / "cat" / Path(cat[1]).name).is_file()
    assert s.tags_of(str(Path(root) / "dog" / Path(cat[0]).name)) == ["dog"]
    assert not any(issues.rglob("*.jpg")) and (issues / "issues_log.csv").is_file()


def test_splits_settings_and_gallery(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.utils import app_settings
    root = tmp_path / "split"
    for sp in ("train", "val"):
        FX.classification(str(root / sp), n=5)
    app_settings().setValue("cls/issues_place", "beside")
    w = _open(shell, qtbot, root)
    s = w.session
    w.open_tool("class_gallery")
    gal = w.open_dialogs["class_gallery"]
    gal.show()
    gal.show_class("cat")
    g = gal.grids[0]
    qtbot.waitUntil(lambda: g.m.rowCount() > 0, timeout=10000)
    n = g.m.rowCount()
    g.setCurrentIndex(g.m.index(0))
    first = [g.m.keys[0], g.m.keys[1], g.m.keys[2]]
    for _ in range(3):                                                # I, I, I: one tile after another
        gal.send_issues()
    assert g.m.rowCount() == n - 3 and not any(k in g.m.pos for k in first)
    beside = (Path(root).resolve().parent / "split_issues")
    sent = sorted(str(p.relative_to(beside)).replace(os.sep, "/") for p in beside.rglob("*.jpg"))
    assert len(sent) == 3 and all(p.split("/")[1] == "cat" and p.split("/")[0] in ("train", "val") for p in sent)
    gal.undo()                                                        # Ctrl+Z in the gallery: the last one back
    qtbot.waitUntil(lambda: g.m.rowCount() == n - 2, timeout=5000)
    assert first[2] in s.by_key and first[2] in g.m.pos
    gal.close()


def test_large_dataset_stays_quick(shell, qtbot, tmp_path):  # noqa: F811
    """10,000 images: setting one aside (the keyboard flow) takes milliseconds, not a rescan."""
    root = tmp_path / "big"
    src = tmp_path / "one.jpg"
    FX._img(str(src), w=32, h=24)
    for c in ("a", "b", "c", "d"):
        (root / c).mkdir(parents=True)
        for i in range(2500):
            os.link(src, root / c / f"{c}_{i:05d}.jpg")
    w = _open(shell, qtbot, root)
    s = w.session
    assert len(s.samples) == 10000
    keys = sorted(s.by_key)[:20]
    t = time.perf_counter()
    for k in keys:
        w.send_to_issues(keys=[k])
    per = (time.perf_counter() - t) / len(keys)
    print(f"send to issues: {per * 1000:.1f} ms per image at 10,000 images")
    assert len(s.samples) == 9980 and per < 0.25, per
    t = time.perf_counter()
    for _ in keys:
        w.inline.undo()
    assert len(s.samples) == 10000 and (time.perf_counter() - t) / len(keys) < 0.25
