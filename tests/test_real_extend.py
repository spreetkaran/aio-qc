# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Extend Model with a REAL model — opt-in (AIOQC_REAL_PYTHON: a Python with torch + ultralytics; YOLO11n is
downloaded the first time). YOLO11n (80 COCO classes) learns a new class, "marker" (a coloured square pasted into
the pictures), from new images labelled ONLY with markers:

  · the enlarged head detects exactly what YOLO11n did before training (old rows copied, not reset)
  · the base model pre-labels the COCO objects on the new images (they would otherwise be learned as background)
  · two stages: the new head with the backbone frozen, then everything at a low learning rate
  · after: the marker is found on the new validation images, and the old classes did not collapse on the old ones
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

REAL = os.environ.get("AIOQC_REAL_PYTHON", "")
pytestmark = [pytest.mark.skipif(not REAL or not os.path.exists(REAL),
                                 reason="set AIOQC_REAL_PYTHON to a Python with torch + ultralytics"),
              pytest.mark.timeout(5400)]

MAKE = r'''
import json, os, random, sys
import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS
old, new, wdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(wdir)
m = YOLO("yolo11n.pt")
names = [m.names[i] for i in range(len(m.names))]
srcs = [cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")]
rng = random.Random(3)

def variant(i):
    im = srcs[i % 2]
    h, w = im.shape[:2]
    s = rng.uniform(0.7, 1.0)
    ch, cw = int(h * s), int(w * s)
    y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
    im = im[y:y + ch, x:x + cw].copy()
    if i % 3 == 1:
        im = cv2.flip(im, 1)
    im = cv2.convertScaleAbs(im, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
    k = 416 / max(im.shape[:2])
    return cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)))

def write(root, split, i, im, lines):
    for d in ("images", "labels"):
        os.makedirs(os.path.join(root, split, d), exist_ok=True)
    cv2.imwrite(os.path.join(root, split, "images", f"img_{i:03d}.jpg"), im)
    open(os.path.join(root, split, "labels", f"img_{i:03d}.txt"), "w").write("".join(lines))

for i in range(30):                                   # the old data: COCO objects, labelled by YOLO11n
    im = variant(i)
    r = m.predict(im, conf=0.5, verbose=False)[0]
    lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for (x, y, w, h), c in
             zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist())]
    write(old, "train" if i < 22 else "val", i, im, lines)
for i in range(40):                                   # the new data: a marker, labelled ONLY with markers
    im = variant(i + 100)
    H, W = im.shape[:2]
    lines = []
    for _ in range(rng.randint(1, 2)):
        s = rng.randint(28, 48)
        x, y = rng.randint(0, W - s - 1), rng.randint(0, H - s - 1)
        cv2.rectangle(im, (x, y), (x + s, y + s), (255, 0, 255), -1)
        cv2.rectangle(im, (x + s // 4, y + s // 4), (x + 3 * s // 4, y + 3 * s // 4), (0, 255, 255), -1)
        lines.append(f"0 {(x + s / 2) / W:.6f} {(y + s / 2) / H:.6f} {s / W:.6f} {s / H:.6f}\n")
    write(new, "train" if i < 30 else "val", i, im, lines)
for root, nm in ((old, names), (new, ["marker"])):
    open(os.path.join(root, "data.yaml"), "w").write(
        f"path: {json.dumps(os.path.abspath(root))}\ntrain: train/images\nval: val/images\nnames:\n" +
        "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(nm)))
print(json.dumps({"names": names, "yolo": os.path.abspath("yolo11n.pt")}))
'''

SAME = r'''
import sys, json
from ultralytics import YOLO
from ultralytics.utils import ASSETS
a, b = YOLO(sys.argv[1]), YOLO(sys.argv[2])
ra, rb = a(str(ASSETS / "bus.jpg"), verbose=False)[0], b(str(ASSETS / "bus.jpg"), verbose=False)[0]
print(json.dumps({"a": [a.names[int(c)] for c in ra.boxes.cls], "b": [b.names[int(c)] for c in rb.boxes.cls],
                  "conf_a": [round(float(x), 3) for x in ra.boxes.conf], "conf_b": [round(float(x), 3) for x in rb.boxes.conf],
                  "nb": len(b.names)}))
'''


def test_yolo11n_learns_a_new_class_and_keeps_coco(tmp_path):
    from AIO_QC.core import extend as X, training as T
    wdir = tmp_path / "w"
    wdir.mkdir()
    (tmp_path / "make.py").write_text(MAKE)
    r = subprocess.run([REAL, str(tmp_path / "make.py"), str(tmp_path / "old"), str(tmp_path / "new"), str(wdir)],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    base_names = info["names"]
    rows, names, old_of, lm = X.plan_classes(base_names, X.yaml_names(tmp_path / "new" / "data.yaml"))
    assert names[:80] == base_names and names[80] == "marker"
    P = X.ExtendPlan(base=info["yolo"], base_names=base_names, names=names, old_of=old_of, label_map=lm,
                     new_yaml=str(tmp_path / "new" / "data.yaml"), old_yaml=str(tmp_path / "old" / "data.yaml"),
                     prelabel=True, review=True, stage1_epochs=4, freeze=10)
    s = {"model": info["yolo"], "epochs": 12, "imgsz": 416, "batch": 8, "device": "cpu", "workers": 0,
         "patience": 50, "project": str(tmp_path / "runs"), "name": "extend", "seed": 0, "plots": False}
    run, st = T.prepare_extend_run(REAL, s, P)
    assert st["images"]["new_train"] == 30 and st["images"]["old_val_old"] == 8 and st["images"]["new_val_new"] == 10
    runner = T.runner_copy()
    spec = run / "aioqc" / "spec.json"

    def go():
        p = subprocess.run([REAL, "-u", str(runner), str(spec)], capture_output=True, text=True, timeout=5000,
                           cwd=str(wdir))
        return p, json.loads((run / "aioqc" / "status.json").read_text())
    # 1 · pre-labels, then it waits for the review
    p, stt = go()
    assert stt["state"] == "review", (stt, p.stdout[-3000:], p.stderr[-3000:])
    pre = json.loads((run / "aioqc" / "prelabels.json").read_text())
    assert pre["added"] > 0                                   # people / a bus on the new images, now labelled
    new_lab = json.loads((run / "aioqc" / "tree.json").read_text())["new_train_labels"][0][1]
    assert any(int(ln.split()[0]) != 80 for ln in Path(new_lab).read_text().splitlines())
    assert all(int(ln.split()[0]) == 0 for ln in (tmp_path / "new" / "train" / "labels" / "img_000.txt")
               .read_text().splitlines())                    # the dataset's own label file is untouched
    # 2 · reviewed: continue — the enlarged head (exactly YOLO11n before training), two stages, before / after
    T.prepare_continue(run)
    p, stt = go()
    assert stt["state"] == "finished", (stt, p.stdout[-3000:], p.stderr[-3000:])
    (tmp_path / "same.py").write_text(SAME)
    q = subprocess.run([REAL, str(tmp_path / "same.py"), info["yolo"], str(run / "aioqc" / "expanded.pt")],
                       capture_output=True, text=True, timeout=600, cwd=str(wdir))
    same = json.loads(q.stdout.strip().splitlines()[-1])
    assert same["a"] == same["b"] and same["nb"] == 81, same
    assert all(abs(x - y) < 0.02 for x, y in zip(same["conf_a"], same["conf_b"])), same
    assert (run / "stage1_head" / "weights" / "best.pt").is_file()
    res = T.extend_result(run)
    assert res is not None, (run / "aioqc" / "train.log")
    new_rows = {x["name"]: x for x in res["rows"] if x["split"] == "val_new"}
    assert (new_rows["marker"]["after"] or 0) > 0.2, new_rows.get("marker")
    old_rows = {x["name"]: x for x in res["rows"] if x["split"] == "val_old"}
    person = old_rows.get("person")
    assert person and person["before"] and person["after"] > person["before"] - 0.15, person
    print(json.dumps({"marker": new_rows["marker"], "person": person, "forgot": res["forgot"]}, indent=1))
