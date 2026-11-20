# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Hand-off batches without the window: export (LabelMe and YOLO, hints), then every merge state — ready, conflict,
unchanged, same, missing, invalid — and the merge itself (backed up; LabelMe keeps the dataset's image fields)."""
import json
from pathlib import Path

import pytest

from AIO_QC.core import handoff as H

from . import fixtures as FX


def _items(root: Path, names, ext=".json"):
    out = []
    for n in names:
        img = next(root.rglob(n + ".jpg"))
        ann = img.with_suffix(ext) if ext == ".json" else img.parent.parent / "labels" / (n + ".txt")
        out.append(H.BatchItem(str(ann), n, str(ann), str(img), note="look at the plate", status="flagged",
                               issues=["Overlap: car / plate"], marks=[(10, 10, 60, 60, "here")]))
    return out


def _edit(p: Path, fn):
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def test_labelme_batch_every_merge_state(tmp_path):
    root = Path(FX.labelme(tmp_path / "ds", n=8))
    names = [f"cam_frame_{i:04d}" for i in range(7)]
    out = tmp_path / "batch"
    st = H.export_batch(root, _items(root, names), out, "labelme", title="B1")
    assert st["items"] == 7 and st["hints"] == 7 and not st["errors"], st
    with pytest.raises(FileExistsError):
        H.export_batch(root, _items(root, names[:1]), out, "labelme")
    assert H.find_batch(out / "batch.json") == out and H.find_batch(tmp_path / "nowhere") is None
    m = H.load_manifest(out)
    assert m["format"] == H.FORMAT and len(m["items"]) == 7 and (out / "notes.csv").is_file()
    ret = {e["ann"].split("/")[-1].split(".")[0]: out / e["batch_ann"] for e in m["items"]}
    ds = {n: root / (n + ".json") for n in names}
    # 0 ready: they relabel, nobody touched it here
    _edit(ret[names[0]], lambda d: d["shapes"][0].update(label="truck"))
    # 1 conflict: they relabel AND it is changed here
    _edit(ret[names[1]], lambda d: d["shapes"][0].update(label="truck"))
    _edit(ds[names[1]], lambda d: d["shapes"].pop())
    # 2 unchanged · 3 same (already has their version)
    _edit(ret[names[3]], lambda d: d["shapes"][0].update(label="van"))
    _edit(ds[names[3]], lambda d: d["shapes"][0].update(label="van"))
    # 4 missing from the returned batch · 5 invalid JSON · 6 different image size
    ret[names[4]].unlink()
    ret[names[5]].write_text("{nope")
    _edit(ret[names[6]], lambda d: d.update(imageWidth=999))
    rows = H.compare(root, m, out)
    state = {Path(r.target).stem: r.state for r in rows}
    assert state == {names[0]: "ready", names[1]: "conflict", names[2]: "unchanged", names[3]: "same",
                     names[4]: "missing", names[5]: "invalid", names[6]: "invalid"}, state
    ready = next(r for r in rows if r.state == "ready")
    assert "truck" in ready.theirs or ready.theirs and ready.choice == "theirs"
    conflict = next(r for r in rows if r.state == "conflict")
    assert conflict.mine and conflict.choice == ""
    conflict.choice = "theirs"
    written, errors, op = H.apply_rows(root, rows, "B1")
    assert len(written) == 2 and not errors and op is not None
    for n in names[:2]:
        d = json.loads(ds[n].read_text())
        assert d["shapes"][0]["label"] == "truck" and d["imagePath"] == n + ".jpg"      # the dataset's own path
    assert H.apply_rows(root, [r for r in rows if r.state == "unchanged"]) == ([], [], None)
    # undo from the backup store puts the dataset files back
    from AIO_QC.core.backups import BackupStore
    store = BackupStore(root)
    try:
        assert store.original_bytes(op, ds[names[0]]) is not None
    finally:
        store.close()


def test_yolo_batch_checks_class_ids(tmp_path):
    root = Path(FX.yolo(tmp_path / "y", n=3))
    items = _items(root, ["train_0", "train_1", "val_2"], ext=".txt")
    out = tmp_path / "b"
    st = H.export_batch(root, items, out, "yolo", names=["car", "plate"], hints=False,
                        class_files=[root / "data.yaml"])
    assert st["items"] == 3 and st["hints"] == 0
    m = H.load_manifest(out)
    assert m["classes"] == ["car", "plate"] and m["kind"] == "yolo"
    rets = [out / e["batch_ann"] for e in m["items"]]
    rets[0].write_text("1 0.5 0.5 0.2 0.2\n")                              # a real change
    rets[1].write_text("7 0.5 0.5 0.2 0.2\n")                              # class 7 does not exist
    rows = H.compare(root, m, out)
    assert [r.state for r in rows] == ["ready", "invalid", "unchanged"], [(r.state, r.reason) for r in rows]
    assert "class" in rows[1].reason
    written, errors, _op = H.apply_rows(root, rows, "y")
    assert len(written) == 1 and (root / "train" / "labels" / "train_0.txt").read_text() == "1 0.5 0.5 0.2 0.2\n"


def test_flat_batch_with_clashing_names_merges_back(tmp_path):
    import shutil
    root = tmp_path / "ds"
    src = Path(FX.labelme(tmp_path / "src", n=2))
    for cam in ("camA", "camB"):
        shutil.copytree(src, root / cam)
    items = []
    for cam in ("camA", "camB"):
        img = root / cam / "cam_frame_0001.jpg"
        ann = img.with_suffix(".json")
        items.append(H.BatchItem(str(ann), f"{cam}/x", str(ann), str(img)))
    out = tmp_path / "flat_batch"
    st = H.export_batch(root, items, out, "labelme", hints=False, flat_naming={"preset": "folder"})
    assert st["items"] == 2
    files = sorted(p.name for p in out.iterdir() if p.suffix in (".json", ".jpg") and p.name != "batch.json"
             and not p.name.startswith("."))
    assert files == ["camA__cam_frame_0001.jpg", "camA__cam_frame_0001.json", "camB__cam_frame_0001.jpg",
                     "camB__cam_frame_0001.json"], files
    assert json.loads((out / "camB__cam_frame_0001.json").read_text())["imagePath"] == "camB__cam_frame_0001.jpg"
    assert "camB/cam_frame_0001.jpg" in (out / "renamed.csv").read_text(encoding="utf-8-sig")
    _edit(out / "camB__cam_frame_0001.json", lambda d: d["shapes"][0].update(label="truck"))
    rows = H.compare(root, H.load_manifest(out), out)
    assert sorted(r.state for r in rows) == ["ready", "unchanged"]
    written, errors, _op = H.apply_rows(root, rows, "flat")
    assert not errors and json.loads((root / "camB" / "cam_frame_0001.json").read_text())["shapes"][0]["label"] == \
        "truck"
    assert json.loads((root / "camB" / "cam_frame_0001.json").read_text())["imagePath"] == "cam_frame_0001.jpg"
