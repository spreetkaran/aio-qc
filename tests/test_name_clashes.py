# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Name Clashes end to end: the same name in two sub-folders (identical and different images), found; renamed in
place with the folder prefix — image and JSON together, imagePath fixed, the review mark and note follow, the cache
follows, renamed.csv written — and Undo puts every name back. Plus YOLO (images/ and labels/ renamed together)."""
import json
import shutil
from pathlib import Path

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _two_cameras(root: Path) -> Path:
    src = Path(FX.labelme(root.parent / "src", n=4))
    for cam in ("camA", "camB"):
        (root / cam).mkdir(parents=True)
        for f in src.iterdir():
            shutil.copy(f, root / cam / f.name)
    FX._img(str(root / "camB" / "cam_frame_0001.jpg"), seed=99)          # same name, a different picture
    return root


def test_find_rename_and_undo(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import REVIEW_FLAG
    from AIO_QC.widgets import name_clashes_dialog as NC
    root = _two_cameras(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    key_b1 = next(k for k in w.session.by_key if "camB" in k and "0001" in k)
    w.session.set_review([key_b1], REVIEW_FLAG, "check the plate")
    w.open_tool("name_clashes")
    d = w.open_dialogs["name_clashes"]
    qtbot.waitUntil(lambda: d.worker is None and "found" in (d.ctx or {}), timeout=30000)
    found = d.ctx["found"]
    same = [f for f in found if f["kind"] == NC.SAME_NAME]
    assert len(same) == 4, found
    ident = {Path(f["items"][0].rel).name: f["identical"] for f in same}
    assert ident["cam_frame_0001.jpg"] is False and ident["cam_frame_0000.jpg"] is True
    assert "Same name in different folders: 4" in d.summarize()
    # rename: the naming panel over the dataset, the folder prefix makes every clashing name unique
    import os
    from AIO_QC.core import renamer as R
    keys = {k for f in same for k in f["keys"]}
    dlg = NC.RenameDialog(d, R.items_for(w.session.samples, os.path.abspath(str(root))), keys,
                          root_len=len(str(root)))
    p = dlg.panel
    p.preset.setCurrentIndex(next(n for n in range(p.preset.count())
                                  if (p.preset.itemData(n) or {}).get("builtin") == "folder"))
    P = p.wait()
    assert P.ok and dlg.ok.isEnabled(), P.problems
    renames = [(it.key, P.new[it.key]) for it in P.items if P.new[it.key] != it.stem]
    assert len(renames) == 8
    dlg.close()
    done, errors = d.run_renames(renames)
    assert len(done) == 8 and not errors
    new_b1 = root / "camB" / "camB__cam_frame_0001.json"
    assert new_b1.is_file() and (root / "camB" / "camB__cam_frame_0001.jpg").is_file()
    assert not (root / "camB" / "cam_frame_0001.json").exists()
    assert json.loads(new_b1.read_text())["imagePath"] == "camB__cam_frame_0001.jpg"
    assert w.session.review.get(str(new_b1))["note"] == "check the plate"            # the mark followed
    from AIO_QC.core.state import state_dir
    csvs = list((state_dir(root) / "renames").glob("renamed_*.csv"))
    assert csvs and "camB/camB__cam_frame_0001.jpg" in csvs[0].read_text(encoding="utf-8-sig")
    qtbot.waitUntil(lambda: d.worker is None, timeout=30000)
    assert not [f for f in d.ctx["found"] if f["kind"] == NC.SAME_NAME]
    # undo: every name back, the mark too
    d.undo_rename()
    assert (root / "camB" / "cam_frame_0001.json").is_file() and not new_b1.exists()
    assert json.loads((root / "camB" / "cam_frame_0001.json").read_text())["imagePath"] == "cam_frame_0001.jpg"
    assert w.session.review.get(str(root / "camB" / "cam_frame_0001.json"))["note"] == "check the plate"
    d.close()


def test_case_only_one_label_and_windows_names(tmp_path):
    from AIO_QC.core import naming as N
    from AIO_QC.widgets import name_clashes_dialog as NC
    for n in ("a/IMG1.jpg", "b/img1.jpg", "c/x.jpg", "c/x.png", "d/con.jpg", "d/bad name.jpg"):
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(n.encode())
    items = [N.Item(str(tmp_path / n), n, path=str(tmp_path / n)) for n in
             ("a/IMG1.jpg", "b/img1.jpg", "c/x.jpg", "c/x.png", "d/con.jpg", "d/bad name.jpg")]
    for it in items:
        it.parent_l = str(Path(it.path).parent).lower()
    kinds = sorted(f["kind"] for f in NC.find(items, len(str(tmp_path))))
    assert kinds == sorted([NC.CASE_ONLY, NC.ONE_LABEL, NC.UNSAFE]), kinds


def test_yolo_image_and_label_renamed_together(shell, qtbot, tmp_path):  # noqa: F811
    root = Path(FX.yolo(tmp_path / "y", n=2))
    w = _open(shell, qtbot, root)
    k = next(k for k in w.session.by_key if "train_0" in k)
    smp = w.session.by_key[k]
    done, errors, log = w.session.rename_samples([(k, "renamed_0")])
    assert done and not errors
    assert (root / "train" / "images" / "renamed_0.jpg").is_file()
    assert (root / "train" / "labels" / "renamed_0.txt").is_file()
    assert not Path(smp.image_path).exists()
    back, errs = w.session.undo_renames(log)
    assert back and not errs and Path(smp.image_path).is_file() and Path(smp.ann_path).is_file()


def test_cached_results_and_backups_follow_a_rename(tmp_path):
    import os
    from collections import Counter
    from AIO_QC.core.backups import BackupStore
    from AIO_QC.core.cache import DatasetCache
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}")
    c = DatasetCache(tmp_path)
    c.put_labels(a, Counter({"car": 2}), None)
    c.flush()
    st = BackupStore(tmp_path)
    op = st.begin("edit", "t")
    st.save(op, a)
    os.replace(a, b)
    assert c.rename([(a, b)]) == 1 and st.rename([(a, b)]) == 1
    assert c.get_labels(b)[0] == Counter({"car": 2}) and st.files(op) == ["b.json"]
    c.close()
    st.close()


def test_convert_to_yolo_flattened_with_unique_names(shell, qtbot, tmp_path):  # noqa: F811
    root = _two_cameras(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    w.open_tool("convert_yolo")
    conv = w.open_dialogs["export"]
    out = tmp_path / "yolo_flat"
    conv.out.setText(str(out))
    conv.flat.setChecked(True)
    conv.folders.setCurrentIndex(conv.folders.findData("flat"))
    P = conv.naming.wait()
    assert P is not None
    if not P.ok:
        conv.naming.fix_remaining()
    assert conv.naming_ready()
    conv.start()
    qtbot.waitUntil(lambda: conv.worker is None, timeout=60000)
    imgs = sorted(p.name for p in (out / "images").iterdir())
    assert all(p.is_file() for p in (out / "images").iterdir())                  # no sub-folders
    assert len(imgs) == 8 and "camA__cam_frame_0001.jpg" in imgs and "camB__cam_frame_0001.jpg" in imgs
    assert sorted(p.stem for p in (out / "labels").iterdir()) == sorted(Path(i).stem for i in imgs)
    csv_text = (out / "renamed.csv").read_text(encoding="utf-8-sig")
    assert "camB/cam_frame_0001.json" in csv_text and "renamed.csv" in (out / "conversion_report.txt").read_text()
    # kept sub-folders (the default) stays as before
    conv.folders.setCurrentIndex(conv.folders.findData("keep"))
    assert conv.options().folders == "keep"
