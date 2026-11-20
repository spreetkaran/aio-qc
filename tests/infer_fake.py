# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A stand-in Python environment for the REAL inference server (core/infer_server.py): `ultralytics` (YOLO + SAM) and
`torch` stand-ins. The toy detector reads the LabelMe JSON next to each image and answers like a model would: the
cars (tightly), the plates on most images, a car nobody labelled on every third image and a plate called a car on
every fifth; a "seg" model draws car outlines. SAM outlines a box as a slightly smaller octagon."""
import stat
import sys
import textwrap
from pathlib import Path

ULTRA = r'''
__version__ = "8.3.0-toy"
import json, os, re
import numpy as np


class _T:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Boxes:
    def __init__(self, rows):
        r = np.asarray(rows, float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls, self.n = _T(r[:, :4]), _T(r[:, 4]), _T(r[:, 5]), len(r)

    def __len__(self):
        return self.n


class _Masks:
    def __init__(self, polys, data=None):
        self.xy = [np.asarray(p, float) for p in polys]
        self.data = _T(data) if data is not None else None


class _R:
    def __init__(self, shape, rows, masks=None, probs=None):
        self.orig_shape = shape
        self.boxes = _Boxes(rows) if probs is None else None
        self.masks = masks
        self.obb = None
        self.keypoints = None
        self.probs = probs


class _Probs:
    def __init__(self, p):
        p = np.asarray(p, float)
        order = np.argsort(-p)
        self.data = _T(p)
        self.top1, self.top1conf = int(order[0]), _T(p[order[0]])
        self.top5 = [int(k) for k in order[:5]]
        self.top5conf = _T(p[order[:5]])


class YOLO:
    def __init__(self, path):
        self.path = str(path)
        self.seg = "seg" in os.path.basename(self.path)
        self.cls = "cls" in os.path.basename(self.path)
        self.task = "classify" if self.cls else "segment" if self.seg else "detect"
        self.names = {0: "cat", 1: "dog", 2: "bird"} if self.cls else {0: "car"} if self.seg else \
            {0: "car", 1: "plate"}
        self.overrides = {"imgsz": 320}

    def _classify(self, p):
        """The class folder (or the flags) the image is in — but every 4th image is called the other class."""
        name = os.path.basename(p)
        i = int(re.search(r"(\d+)\.", name).group(1))
        truth = os.path.basename(os.path.dirname(p))
        js = os.path.splitext(p)[0] + ".json"
        if os.path.isfile(js):
            fl = json.load(open(js)).get("flags") or {}
            truth = next((k for k, v in fl.items() if v is True), truth)
        ids = {v: k for k, v in self.names.items()}
        k = ids.get(truth, 2)
        if i % 4 == 0:
            k = 1 - k if k < 2 else 0
        probs = [0.04, 0.04, 0.04]
        probs[k] = 0.92
        import cv2
        im = cv2.imread(p)
        return _R(im.shape[:2] if im is not None else (240, 320), [], probs=_Probs(probs))

    def _one(self, p, kw):
        if isinstance(p, np.ndarray):
            return _R(p.shape[:2], [], probs=_Probs([0.5, 0.3, 0.2]) if self.cls else None)
        if self.cls:
            return self._classify(p)
        d = json.load(open(os.path.splitext(p)[0] + ".json"))
        i = int(re.search(r"_(\d+)\.", os.path.basename(p)).group(1))
        rows, polys = [], []
        for sh in d["shapes"]:
            pts = np.asarray(sh["points"], float)
            if sh["shape_type"] == "circle" or len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
            if sh["label"] == "car":
                rows.append([x1 + 1, y1 + 1, x2 - 1, y2 - 1, 0.92, 0])
                polys.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [(x1 + x2) / 2, y2 + 2]])
            elif sh["label"] == "plate" and not self.seg and i % 4:
                rows.append([x1, y1, x2, y2, 0.76, 1])
        if not self.seg:
            if i % 3 == 0:
                rows.append([250, 10, 300, 60, 0.81, 0])                    # an object nobody labelled
            if i % 5 == 0:
                plate = next((s for s in d["shapes"] if s["label"] == "plate"), None)
                if plate is not None:
                    pts = np.asarray(plate["points"], float)
                    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
                    rows.append([x1, y1, x2, y2, 0.66, 0])                  # the plate called a car
        conf, keep = float(kw.get("conf", 0.25)), kw.get("classes")
        ok = [n for n, r in enumerate(rows) if r[4] >= conf and (keep is None or int(r[5]) in keep)]
        rows = [rows[n] for n in ok]
        masks = _Masks([polys[n] for n in ok]) if self.seg else None
        return _R((d["imageHeight"], d["imageWidth"]), rows, masks)

    def predict(self, src, **kw):
        srcs = src if isinstance(src, list) else [src]
        return [self._one(p, kw) for p in srcs]

    def export(self, **kw):
        out = os.path.splitext(self.path)[0] + "." + kw.get("format", "onnx")
        open(out, "wb").write(b"toy")
        return out


class SAM:
    def __init__(self, path):
        self.path = path

    def __call__(self, image, bboxes=None, verbose=False):
        import cv2
        im = cv2.imread(image)
        h, w = im.shape[:2]
        data = []
        for x1, y1, x2, y2 in bboxes or []:
            m = np.zeros((h, w), np.uint8)
            cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2 - 2, (y2 - y1) / 2 - 2
            pts = np.array([[cx + rx * np.cos(a), cy + ry * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]],
                           np.int32)
            cv2.fillPoly(m, [pts], 1)
            data.append(m)
        return [_R((h, w), [], _Masks([], np.stack(data) if data else np.zeros((0, h, w))))]
'''

TORCH = '''
__version__ = "2.9-toy"
class cuda:
    @staticmethod
    def is_available():
        return False
class backends:
    mps = None
'''


def fake_python(tmp: Path) -> str:
    site = tmp / "infer_site"
    (site / "ultralytics").mkdir(parents=True, exist_ok=True)
    (site / "ultralytics" / "__init__.py").write_text(textwrap.dedent(ULTRA))
    (site / "torch").mkdir(exist_ok=True)
    (site / "torch" / "__init__.py").write_text(TORCH)
    py = tmp / "inferpython"
    py.write_text(f"#!/bin/sh\nPYTHONPATH={site} exec {sys.executable} \"$@\"\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return str(py)


def models_config(root: Path, tmp: Path, seg: bool = True):
    from AIO_QC.core.state import save_models_config
    det = tmp / "toy_det.pt"
    det.write_bytes(b"det")
    models = [{"id": "md", "name": "toy det", "path": str(det), "enabled": True, "imgsz": 320, "task": "detect",
               "names": {"0": "car", "1": "plate"}, "class_map": {"0": "car", "1": "plate"}}]
    if seg:
        s = tmp / "toy_seg.pt"
        s.write_bytes(b"seg")
        models.append({"id": "ms", "name": "toy seg", "path": str(s), "enabled": True, "imgsz": 320,
                       "task": "segment", "names": {"0": "car"}, "class_map": {"0": "car"}})
    save_models_config(root, {"version": 1, "models": models})
    return models
