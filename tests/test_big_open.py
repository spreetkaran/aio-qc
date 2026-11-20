# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Opening big datasets stays responsive: format detection never reads a LabelMe folder's JSONs (it read every one —
gigabytes on a 12 GB dataset — to rule out COCO, with the window frozen), the file list is not flooded with per-row
signals while indexing, and the YOLO class-id scan stops when the dataset closes."""
import json
import threading
from pathlib import Path


def _labelme(root: Path, n: int):
    root.mkdir(parents=True, exist_ok=True)
    blob = "A" * 50_000                                   # an embedded image, as LabelMe saves it
    for i in range(n):
        (root / f"img_{i:04d}.json").write_text(json.dumps({"version": "5.4.1", "flags": {}, "shapes": [
            {"label": "car", "points": [[1, 1], [5, 5]], "shape_type": "rectangle"}],
            "imagePath": f"img_{i:04d}.jpg", "imageData": blob, "imageHeight": 8, "imageWidth": 8}))


def test_coco_probe_is_cheap_and_still_finds_coco(tmp_path, monkeypatch):
    import AIO_QC.core.coco_dataset as C
    from AIO_QC.core.dataset import DatasetType, detect_dataset_types
    root = tmp_path / "lm"
    _labelme(root, 200)
    calls, read = [], [0]
    real_is, real_open = C.is_coco_file, open

    def counting(p):
        calls.append(p)
        return real_is(p)
    monkeypatch.setattr(C, "is_coco_file", counting)
    assert C.find_coco_files(root) == [] and len(calls) <= C.MAX_PROBES_PER_DIR
    # a LabelMe file is ruled out from its first 4 KB
    class F:
        def __init__(self, *a, **k):
            self.f = real_open(*a, **k)

        def read(self, n=-1):
            d = self.f.read(n)
            read[0] += len(d)
            return d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()
    monkeypatch.setattr("builtins.open", F)
    assert real_is(root / "img_0000.json") is False and read[0] <= 4096
    monkeypatch.setattr("builtins.open", real_open)
    assert detect_dataset_types(root) == [DatasetType.LABELME]
    # a real COCO file among many LabelMe JSONs, and a Roboflow-style one next to images, are still found
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 8, "height": 8}], "annotations": [],
            "categories": [{"id": 1, "name": "car"}]}
    (root / "instances_train.json").write_text(json.dumps(coco))
    (tmp_path / "rf" / "train").mkdir(parents=True)
    (tmp_path / "rf" / "train" / "_annotations.coco.json").write_text(json.dumps(coco))
    calls.clear()
    assert [p.name for p in C.find_coco_files(root)] == ["instances_train.json"]
    assert [p.name for p in C.find_coco_files(tmp_path / "rf")] == ["_annotations.coco.json"]


def test_index_flags_signal_only_changes(qtbot):
    from AIO_QC.core.dataset import Sample
    from AIO_QC.widgets.file_list import SampleListModel
    m = SampleListModel()
    m.set_samples([Sample(key=f"k{i}", name=f"k{i}", image_path=None, ann_path=None) for i in range(5000)])
    got = []
    m.dataChanged.connect(lambda a, b, roles: got.append((a.row(), b.row())))
    m.set_flags({f"k{i}": None for i in range(5000)})            # a clean dataset: nothing changes, nothing sent
    assert got == []
    m.set_flags({f"k{i}": "empty" for i in range(1000, 3000)})   # many: one range
    assert got == [(1000, 2999)] and m.flags["k1500"] == "empty"
    got.clear()
    m.set_flags({"k1500": None, "k7": "error"})                  # a few: per row
    assert sorted(got) == [(7, 7), (1500, 1500)]


def test_class_id_scan_stops(tmp_path):
    from AIO_QC.core.class_check import scan_class_ids
    files = []
    for i in range(3000):
        p = tmp_path / f"{i}.txt"
        p.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
        files.append(p)
    assert scan_class_ids(files, workers=2).files == 3000
    ev = threading.Event()
    ev.set()
    out = scan_class_ids(files, workers=2, stop=ev.is_set)
    assert out.cancelled and out.files < 3000
