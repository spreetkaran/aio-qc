# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The new model features with REAL models — opt-in, because it needs a Python environment with PyTorch and Ultralytics
and downloads YOLO11n and MobileSAM (about 45 MB) the first time:

    Windows   set AIOQC_REAL_PYTHON=C:\\path\\to\\env\\python.exe
              python -m pytest tests\\test_real_models.py -v
    Linux     AIOQC_REAL_PYTHON=/path/to/env/bin/python python -m pytest tests/test_real_models.py -v

A small dataset is made from the two photos Ultralytics ships (bus.jpg, zidane.jpg — cropped, flipped, lit
differently), labelled by YOLO11n, with planted mistakes: a person left unlabelled, a person labelled "bus". Then:
Model Check finds the planted mistakes · Model Assist suggests the missing person and it is accepted · SAM outlines a
box · Convert to YOLO · Train Model trains for real (1 epoch, CPU is fine) · Improve Model checks the new model, flags
the hardest images, picks new images from a pool where it is unsure, and sets up fine-tuning.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from .test_aio_image import _open, shell  # noqa: F401

REAL = os.environ.get("AIOQC_REAL_PYTHON", "")
pytestmark = [pytest.mark.skipif(not REAL or not os.path.exists(REAL),
                                 reason="set AIOQC_REAL_PYTHON to a Python with torch + ultralytics"),
              pytest.mark.timeout(3600)]

MAKE = r'''
import json, os, random, sys
import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS
out, pool, wdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(wdir)
m = YOLO("yolo11n.pt")
names = {int(k): v for k, v in m.names.items()}
srcs = [cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")]
rng = random.Random(7)
planted = {"missing": [], "wrong": []}

def variant(i):
    im = srcs[i % 2]
    h, w = im.shape[:2]
    s = rng.uniform(0.75, 1.0)
    ch, cw = int(h * s), int(w * s)
    y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
    im = im[y:y + ch, x:x + cw]
    if i % 3 == 1:
        im = cv2.flip(im, 1)
    im = cv2.convertScaleAbs(im, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
    k = 480 / max(im.shape[:2])
    return cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)))

for n, folder in ((24, out), (12, pool)):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        im = variant(i + (0 if folder == out else 100))
        name = f"img_{i:03d}.jpg"
        p = os.path.join(folder, name)
        cv2.imwrite(p, im)
        if folder == pool:
            continue
        r = m.predict(p, conf=0.5, classes=[0, 5], verbose=False)[0]
        shapes = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            shapes.append({"label": names[int(c)], "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle",
                           "flags": {}, "group_id": None, "description": ""})
        persons = [s for s in shapes if s["label"] == "person"]
        if i % 6 == 1 and persons:
            shapes.remove(persons[0])
            planted["missing"].append(name)
        elif i % 6 == 3 and persons:
            persons[0]["label"] = "bus"
            planted["wrong"].append(name)
        json.dump({"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": name, "imageData": None,
                   "imageHeight": im.shape[0], "imageWidth": im.shape[1]},
                  open(os.path.join(folder, name[:-4] + ".json"), "w"), indent=1)
SAM_W = "mobile_sam.pt"
from ultralytics import SAM
SAM(SAM_W)
print(json.dumps({"planted": planted, "names": {str(k): v for k, v in names.items()},
                  "yolo": os.path.abspath("yolo11n.pt"), "sam": os.path.abspath(SAM_W)}))
'''


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("real")
    wdir = base / "weights"
    wdir.mkdir()
    (base / "make.py").write_text(MAKE)
    r = subprocess.run([REAL, str(base / "make.py"), str(base / "ds"), str(base / "pool"), str(wdir)],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["planted"]["missing"] and info["planted"]["wrong"]
    return base, info


def _labels(root, name):
    return [s["label"] for s in json.loads((root / name).read_text())["shapes"]]


def test_real_models_end_to_end(world, shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    from AIO_QC.core import improve as IM, model_registry as R
    from AIO_QC.core import training as T
    from AIO_QC.core.model_check import MISMATCH, MISSING
    from PyQt6.QtWidgets import QMessageBox
    base, info = world
    root = base / "ds"
    w = _open(shell, qtbot, root)
    w.inference.set_python(REAL)
    w.settings.setValue("train/python", REAL)
    entry = R.register(info["yolo"], root=w.session.root, name="YOLO11n", names=info["names"], task="detect",
                       labels=set(w.session.label_totals()))
    assert entry["class_map"].get("0") == "person" and entry["class_map"].get("5") == "bus"

    # 1 · Model Check finds the planted mistakes
    w.open_tool("model_check")
    mc = w.open_dialogs["model_check"]
    mc.start_scan()
    qtbot.waitUntil(lambda: mc.worker is None, timeout=900000)
    assert mc.ctx.get("eval") is not None, (mc.status.text(), mc.errors[:3])
    found = {(r.cells[0], Path(r.key).name.replace(".json", ".jpg")) for r in mc.rows if r is not None}
    miss = [n for n in info["planted"]["missing"] if (MISSING, n) in found]
    wrong = [n for n in info["planted"]["wrong"] if (MISMATCH, n) in found]
    assert len(miss) >= len(info["planted"]["missing"]) - 1, (info["planted"], sorted(found))
    assert len(wrong) >= len(info["planted"]["wrong"]) - 1, (info["planted"], sorted(found))
    mc.close()

    # 2 · Model Assist suggests the missing person on its image; accepted, it is in the file
    target = miss[0].replace(".jpg", ".json")
    w.show_sample(str(root / target))
    qtbot.waitUntil(lambda: w.current_key == str(root / target), timeout=10000)
    before = _labels(root, target).count("person")
    w.assist.suggest()
    qtbot.waitUntil(lambda: bool(w.assist.items or w.assist.last_error), timeout=300000)
    assert not w.assist.last_error, w.assist.last_error
    i = next(n for n, x in enumerate(w.assist.items) if x["label"] == "person")
    w.assist.accept(i)
    assert _labels(root, target).count("person") == before + 1

    # 3 · SAM (MobileSAM) outlines a suggestion box for real
    w.settings.setValue("assist/sam_model", info["sam"])
    w.settings.setValue("assist/sam", True)
    other = next(n for n in sorted(os.listdir(root)) if n.endswith(".json") and n != target)
    w.show_sample(str(root / other))
    qtbot.waitUntil(lambda: w.current_key == str(root / other), timeout=10000)
    shapes0 = json.loads((root / other).read_text())["shapes"]
    from PyQt6.QtCore import QRectF
    first = next(s for s in shapes0 if s["shape_type"] == "rectangle")
    (x1, y1), (x2, y2) = first["points"]
    monkeypatch.setattr(w.inline, "_choose_label", lambda *a, **k: first["label"])
    w.inline.on_box(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
    qtbot.waitUntil(lambda: len(json.loads((root / other).read_text())["shapes"]) == len(shapes0) + 1,
                    timeout=600000)
    new = json.loads((root / other).read_text())["shapes"][-1]
    assert new["shape_type"] == "polygon" and len(new["points"]) >= 4, (new, w.statusBar().currentMessage())
    w.settings.setValue("assist/sam", False)

    # 4 · Convert to YOLO with a split, then train for real
    w.open_tool("convert_yolo")
    cv = w.open_dialogs["export"]
    yolo = base / "yolo"
    cv.out.setText(str(yolo))
    cv.start()
    qtbot.waitUntil(lambda: cv.worker is None, timeout=600000)
    assert (yolo / "dataset.yaml").is_file(), cv.status.text()
    w.open_dialogs["export"].close()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w.open_training(yaml=yolo / "dataset.yaml")
    td = w.train_dialog
    qtbot.waitUntil(lambda: td.check is not None and td.env is not None and not td.workers, timeout=900000)
    i = td.model_box.findData(info["yolo"])
    if i < 0:
        td.model_box.insertItem(0, "yolo11n.pt", info["yolo"])
        i = 0
    td.model_box.setCurrentIndex(i)
    td.epochs.setValue(1)
    td.imgsz.setValue(160)
    td.project.setText(str(base / "runs"))
    td.start()
    run = td.current
    assert run is not None
    qtbot.waitUntil(lambda: T.state_of(T.status(run)) in ("finished", "failed", "stopped"), timeout=3000000)
    assert T.state_of(T.status(run)) == "finished", (run / "aioqc" / "train.log").read_text()[-3000:] \
        if (run / "aioqc" / "train.log").exists() else T.status(run)
    assert (run / "weights" / "best.pt").is_file()

    # 5 · Improve Model on the trained model: check, flag, pick, fine-tune
    d = w.open_improve(run=run, check=True)
    qtbot.waitUntil(lambda: (IM.load(w.session.root).get("history") or []) != [] or "Not checked" in
                    d.check_note.text(), timeout=900000)
    st = IM.load(w.session.root)
    assert st.get("history"), d.check_note.text()
    assert d.classes.rowCount() >= 1 and st["last_check"]["spots"]["classes"]
    d.n_hard.setValue(5)
    d.flag_hardest()
    assert "flagged" in d.label_note.text()
    d.pick()
    from AIO_QC.widgets.data_prep import PickPage
    dp = w.open_dialogs["data_prep"]
    pg = dp.page()
    assert isinstance(pg, PickPage)
    pg.src.setText(str(base / "pool"))
    pg.n.setValue(4)
    dp._preview()
    qtbot.waitUntil(lambda: dp.job is None, timeout=900000)
    dp._run()
    qtbot.waitUntil(lambda: dp.job is None, timeout=900000)
    out = Path(pg.out.text())
    picked = sorted(out.glob("*.jpg"))
    assert len(picked) == 4 and all(p.with_suffix(".json").is_file() for p in picked), dp.status.text()
    assert all(str(p.with_suffix(".json")) in w.session.by_key for p in picked)          # joined the dataset
    d.retrain()
    assert w.train_dialog.current == run and w.train_dialog.settings_dict()["lr0"] > 0
    d.close()
    td.close()
