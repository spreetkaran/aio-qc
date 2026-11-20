# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Classifiers with a REAL model — opt-in (AIOQC_REAL_PYTHON: a Python with torch + ultralytics; YOLO11n-cls is
downloaded the first time). On pictures of red and blue squares:

  · Train Model trains YOLO11n-cls from a class-folder dataset (train/ val/) and reports per-class accuracy
  · the inference server returns its top classes (what Model Check and Model Assist use)
  · Extend Model adds a third class (green) — the enlarged last layer predicts exactly what the old one did, and
    after training the new class is learned while red / blue keep their accuracy
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


def _make(root: Path, classes, n_train=24, n_val=8, seed=0):
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    col = {"red": (40, 40, 220), "blue": (220, 60, 40), "green": (40, 200, 40)}
    for sp, n in (("train", n_train), ("val", n_val)):
        for c in classes:
            d = root / sp / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                im = rng.integers(90, 160, (96, 96, 3), dtype=np.uint8)
                s = int(rng.integers(30, 60))
                x, y = int(rng.integers(0, 96 - s)), int(rng.integers(0, 96 - s))
                cv2.rectangle(im, (x, y), (x + s, y + s), col[c], -1)
                cv2.imwrite(str(d / f"{c}_{i}.jpg"), im)
    return root


def _run(spec_path, cwd):
    from AIO_QC.core import training as T
    runner = T.runner_copy()
    p = subprocess.run([REAL, "-u", str(runner), str(spec_path)], capture_output=True, text=True, timeout=5000,
                       cwd=str(cwd))
    return p, json.loads((Path(spec_path).parent / "status.json").read_text())


SAME = r'''
import sys, json
from ultralytics import YOLO
a, b = YOLO(sys.argv[1]), YOLO(sys.argv[2])
out = []
for p in sys.argv[3:]:
    ra, rb = a(p, verbose=False)[0], b(p, verbose=False)[0]
    out.append([a.names[int(ra.probs.top1)], b.names[int(rb.probs.top1)], round(float(ra.probs.top1conf), 3),
                round(float(rb.probs.top1conf), 3)])
print(json.dumps(out))
'''


def test_train_and_extend_a_classifier(tmp_path):
    from AIO_QC.core import extend as X, training as T
    wdir = tmp_path / "w"
    wdir.mkdir()
    old = _make(tmp_path / "old", ["blue", "red"])
    chk = T.check_dataset(old)
    assert chk.task == "cls" and chk.ok, chk.issues
    s = {"model": "yolo11n-cls.pt", "epochs": 6, "imgsz": 64, "batch": 16, "device": "cpu", "workers": 0,
         "patience": 50, "project": str(tmp_path / "runs"), "name": "base", "seed": 0}
    run = T.prepare_run(REAL, chk, s)
    p, st = _run(run / "aioqc" / "spec.json", wdir)
    assert st["state"] == "finished", (st, p.stdout[-3000:], p.stderr[-3000:])
    per = {r["name"]: r for r in st.get("per_class") or []}
    assert set(per) == {"blue", "red"} and min(r["box_r"] for r in per.values()) > 0.8, per
    best = st["best"]
    # Extend: + green
    new = _make(tmp_path / "new", ["green"], n_train=24, n_val=8, seed=5)
    info = X.base_info(best)
    assert info["names"] == ["blue", "red"] and info["dataset_yaml"] == str(old)
    rows, names, old_of, lm = X.plan_classes(info["names"], X.yaml_names(new))
    P = X.ExtendPlan(base=best, base_names=info["names"], names=names, old_of=old_of, label_map=lm,
                     new_yaml=str(new), old_yaml=str(old), stage1_epochs=3, freeze=8, prelabel=False, review=False)
    s2 = dict(s, model=best, epochs=6, name="extend")
    run2, _st = T.prepare_extend_run(REAL, s2, P)
    p, st2 = _run(run2 / "aioqc" / "spec.json", wdir)
    assert st2["state"] == "finished", (st2, p.stdout[-3000:], p.stderr[-3000:])
    (tmp_path / "same.py").write_text(SAME)
    imgs = [str(x) for x in sorted((old / "val" / "red").glob("*.jpg"))[:3] + sorted((old / "val" / "blue")
                                                                                     .glob("*.jpg"))[:3]]
    q = subprocess.run([REAL, str(tmp_path / "same.py"), best, str(run2 / "aioqc" / "expanded.pt"), *imgs],
                       capture_output=True, text=True, timeout=600, cwd=str(wdir))
    same = json.loads(q.stdout.strip().splitlines()[-1])
    assert all(a == b and abs(ca - cb) < 0.05 for a, b, ca, cb in same), same     # old rows copied by name
    res = T.extend_result(run2)
    rows = {(r["split"], r["name"]): r for r in res["rows"]}
    assert rows[("val_new", "green")]["after"] > 0.7, rows
    assert not res["forgot"], res["forgot"]
    print(json.dumps({k[0] + "/" + k[1]: (v["before"], v["after"]) for k, v in rows.items()}, indent=1))
