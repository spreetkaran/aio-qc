# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Small synthetic datasets for the tests (LabelMe with video frames + text, YOLO splits, classification, tables)."""
from __future__ import annotations

import json
import os

import numpy as np


def _img(path, w=320, h=240, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    a = (rng.random((h, w, 3)) * 60 + 80).astype(np.uint8)
    cv2.rectangle(a, (40 + seed % 50, 40), (140 + seed % 50, 120), (30, 200, 60), -1)
    cv2.imwrite(str(path), a)


def labelme(root, n=12):
    os.makedirs(root, exist_ok=True)
    texts = ["MSKU1234565"] * (n - 3) + ["MSKU1234S65", "", "mskU 123456 5"]
    for i in range(n):
        name = f"cam_frame_{i:04d}"
        _img(os.path.join(root, name + ".jpg"), seed=i)
        x = 40 + i % 50
        shapes = [
            {"label": "car", "points": [[x, 40], [x + 100, 120]], "shape_type": "rectangle", "flags": {},
             "group_id": None, "description": ""},
            {"label": "plate", "points": [[x + 10, 90], [x + 60, 85], [x + 62, 100], [x + 12, 105]],
             "shape_type": "polygon", "flags": {}, "group_id": None, "description": texts[i]},
            {"label": "wheel", "points": [[x + 20, 120], [x + 30, 120]], "shape_type": "circle", "flags": {},
             "group_id": None, "description": ""},
        ]
        if i == 3:
            shapes.append({"label": "Car ", "points": [[5, 5], [9, 9]], "shape_type": "rectangle", "flags": {},
                           "group_id": None, "description": ""})
        json.dump({"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": name + ".jpg", "imageData": None,
                   "imageHeight": 240, "imageWidth": 320}, open(os.path.join(root, name + ".json"), "w"))
    return root


def yolo(root, n=8):
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)
        for i in range(n):
            _img(os.path.join(root, split, "images", f"{split}_{i}.jpg"), seed=i + (100 if split == "val" else 0))
            with open(os.path.join(root, split, "labels", f"{split}_{i}.txt"), "w") as f:
                f.write(f"0 0.3 0.33 0.3 0.33\n1 0.6 0.6 0.1 0.08\n")
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: car\n  1: plate\n")
    return root


def classification(root, n=5):
    for c in ("cat", "dog"):
        os.makedirs(os.path.join(root, c), exist_ok=True)
        for i in range(n):
            _img(os.path.join(root, c, f"{c}_{i}.jpg"), seed=i + (7 if c == "dog" else 0))
    return root


def table(path, n=500):
    import pandas as pd
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(n), "city": rng.choice(["Pune", "Delhi", " Mumbai ", None], n),
                       "amount": np.round(rng.lognormal(4, 1, n), 2), "qty": rng.integers(1, 9, n).astype(str),
                       "when": pd.date_range("2024-01-01", periods=n, freq="h").astype(str)})
    df.to_csv(path, index=False)
    return path
