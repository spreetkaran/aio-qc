# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
YOLO detectors for model-assisted mining and pre-annotation (Qt-free).

Backends (chosen by file extension / availability):
  .onnx         -> onnxruntime (CUDA / DirectML / CPU providers) in this process, else cv2.dnn
  .pt / .engine -> a Python environment that has ultralytics + torch (conda / venv / temporary env),
                   served by core/infer_server.py in a subprocess (AIO QC's approach): the app never
                   imports torch, any env works, a CUDA crash can't take the GUI down, models stay loaded.
                   Falls back to an in-process ultralytics if it is importable and no env is set.

Output layouts handled automatically:
  YOLOv8/v11 (ultralytics)  (1, 4+nc[+masks], N)   boxes cx,cy,w,h + class scores
  YOLOv5/v7                 (1, N, 5+nc)           boxes + objectness + class scores
  End-to-end / NMS-free     (1, <=1000, 6)         x1,y1,x2,y2,score,class  (YOLOv10, nms=True exports)

Class names are read from ultralytics ONNX metadata; otherwise from a sidecar
file next to the model (<model>.names / .txt / .yaml / data.yaml / classes.txt).
"""
import ast
import os
from dataclasses import dataclass

import cv2
import numpy as np

from AIO_QC.tools.frame_extractor.core.imageio import imread

try:
    import onnxruntime as ort
    HAVE_ORT = True
except Exception:  # pragma: no cover
    ort = None
    HAVE_ORT = False


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    name: str
    model: str = ""

    def iou(self, o):
        ix1, iy1 = max(self.x1, o.x1), max(self.y1, o.y1)
        ix2, iy2 = min(self.x2, o.x2), min(self.y2, o.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a = (self.x2 - self.x1) * (self.y2 - self.y1)
        b = (o.x2 - o.x1) * (o.y2 - o.y1)
        return inter / (a + b - inter + 1e-9)


# ------------------------------------------------------------------ names
def _parse_names(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw)}
    try:
        return _parse_names(ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return None


def _sidecar_names(model_path):
    base, _ = os.path.splitext(model_path)
    d = os.path.dirname(model_path)
    for p in (base + ".names", base + ".txt", base + ".yaml", os.path.join(d, "data.yaml"),
              os.path.join(d, "classes.txt")):
        if not os.path.isfile(p):
            continue
        text = open(p, encoding="utf-8").read()
        if p.endswith(".yaml"):
            # minimal YAML: "names: [a, b]" or "names:\n  0: a\n  1: b" / "  - a"
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("names:"):
                    rest = line.split(":", 1)[1].strip()
                    if rest:
                        names = _parse_names(rest)
                        if names:
                            return names
                    out = {}
                    for sub in lines[i + 1:]:
                        if not sub.startswith((" ", "\t", "-")):
                            break
                        sub = sub.strip()
                        if sub.startswith("- "):
                            out[len(out)] = sub[2:].strip().strip("'\"")
                        elif ":" in sub:
                            k, v = sub.split(":", 1)
                            out[int(k)] = v.strip().strip("'\"")
                    if out:
                        return out
        else:
            names = [l.strip() for l in text.splitlines() if l.strip()]
            if names:
                return {i: n for i, n in enumerate(names)}
    return None


# ----------------------------------------------------------- pre/post-proc
def letterbox(img, size):
    """Resize keeping aspect and pad (114) to size=(h, w). Returns img, scale, (padx, pady)."""
    h, w = img.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (tw - nw) / 2, (th - nh) / 2
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out = cv2.copyMakeBorder(img, top, th - nh - top, left, tw - nw - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def _nms(boxes_xyxy, scores, cls, iou):
    if len(scores) == 0:
        return []
    xywh = np.column_stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                            boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]])
    if hasattr(cv2.dnn, "NMSBoxesBatched"):
        keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.astype(int).tolist(), 0.0, iou)
    else:  # class-offset trick
        off = xywh.copy()
        off[:, :2] += cls[:, None] * 10000.0
        keep = cv2.dnn.NMSBoxes(off.tolist(), scores.tolist(), 0.0, iou)
    return np.array(keep).reshape(-1).astype(int).tolist()


def decode_output(out, nc, conf, iou):
    """Raw model output -> (boxes_xyxy [in letterboxed input pixels], scores, cls)."""
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    a, b = out.shape
    # End-to-end (NMS already applied)
    if b == 6 and a <= 1000 and not (nc == 1 and a > 1000):
        det = out[out[:, 4] >= conf]
        return det[:, :4], det[:, 4], det[:, 5].astype(int), True
    if a < b:                      # (4+nc[+nm], N) -> (N, C)   YOLOv8/v11
        out = out.T
        c = out.shape[1]
        nc = nc or (c - 4)
        scores_all = out[:, 4:4 + nc]
        obj = None
    else:                          # (N, 5+nc)  YOLOv5/v7   (or (N, 4+nc) if nc known)
        c = out.shape[1]
        if nc and c == 4 + nc:
            scores_all, obj = out[:, 4:4 + nc], None
        else:
            nc = nc or (c - 5)
            obj = out[:, 4:5]
            scores_all = out[:, 5:5 + nc] * obj
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    m = scores >= conf
    xywh, scores, cls = out[m, :4], scores[m], cls[m]
    boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                             xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])
    keep = _nms(boxes, scores, cls, iou)
    return boxes[keep], scores[keep], cls[keep], False


def dets_to_array(dets):
    """Detection list -> float32 (n, 6): x1, y1, x2, y2, conf, cls."""
    if not dets:
        return np.zeros((0, 6), np.float32)
    return np.array([[d.x1, d.y1, d.x2, d.y2, d.conf, d.cls] for d in dets], np.float32)


def _rows(boxes):
    return np.asarray(boxes or [], np.float32).reshape(-1, 6)


class YoloDetector:
    max_det = 300

    def __init__(self, path, conf=0.25, iou=0.45, imgsz=0, device="auto", threads=0):
        self.path = path
        self.label = os.path.splitext(os.path.basename(path))[0]
        self.conf, self.iou = conf, iou
        self.names = None
        self.imgsz = (imgsz, imgsz) if imgsz else None
        self._ul = None
        self._sess = self._net = None
        self.provider = "cpu"
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pt":
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(f"{os.path.basename(path)}: .pt needs `pip install ultralytics`, "
                                  f"or export once: yolo export model={os.path.basename(path)} format=onnx")
            self._ul = YOLO(path)
            self.names = _parse_names(self._ul.names)
            self.imgsz = self.imgsz or (640, 640)
            self.provider = "ultralytics"
        elif ext == ".onnx":
            if HAVE_ORT:
                self._init_ort(device, threads)
            else:
                self._net = cv2.dnn.readNetFromONNX(path)
                self.provider = "cv2.dnn"
        else:
            raise ValueError(f"Unsupported model file: {path} (use .onnx or .pt)")
        self.names = self.names or _sidecar_names(path)
        self.imgsz = self.imgsz or (640, 640)

    def _init_ort(self, device, threads):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = int(threads)
        avail = ort.get_available_providers()
        prefer = [] if device == "cpu" else [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                                         "CoreMLExecutionProvider") if p in avail]
        self._sess = ort.InferenceSession(self.path, opts, providers=prefer + ["CPUExecutionProvider"])
        self.provider = self._sess.get_providers()[0].replace("ExecutionProvider", "").lower()
        inp = self._sess.get_inputs()[0]
        self.dynamic_batch = not isinstance(inp.shape[0], int)
        self._in_name = inp.name
        self._fp16 = "float16" in inp.type
        meta = self._sess.get_modelmeta().custom_metadata_map
        self.names = _parse_names(meta.get("names"))
        if not self.imgsz:
            shp = inp.shape
            if isinstance(shp[2], int) and isinstance(shp[3], int):
                self.imgsz = (shp[2], shp[3])
            elif meta.get("imgsz"):
                v = ast.literal_eval(meta["imgsz"])
                self.imgsz = tuple(v) if isinstance(v, (list, tuple)) else (v, v)

    def name_of(self, c):
        return self.names.get(int(c), str(int(c))) if self.names else str(int(c))

    def _postprocess(self, out, conf, r, pad, w, h):
        nc = len(self.names) if self.names else 0
        boxes, scores, cls, _ = decode_output(out, nc, conf, self.iou)
        if len(scores) > self.max_det:
            top = np.argsort(-scores)[:self.max_det]
            boxes, scores, cls = boxes[top], scores[top], cls[top]
        px, py = pad
        b = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = ((b[:, [0, 2]] - px) / r).clip(0, w)
        b[:, [1, 3]] = ((b[:, [1, 3]] - py) / r).clip(0, h)
        keep = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
        return np.column_stack([b[keep], np.asarray(scores, np.float32)[keep],
                                np.asarray(cls, np.float32)[keep]]).astype(np.float32).reshape(-1, 6)

    def detect_batch(self, frames, conf=None):
        """List of BGR frames -> list of (n, 6) float32 arrays (model class ids). One forward pass per batch
        when the ONNX model has a dynamic batch axis (or ultralytics in-process); else frame by frame."""
        conf = self.conf if conf is None else conf
        if self._ul is not None:
            res = self._ul.predict(list(frames), conf=conf, iou=self.iou, imgsz=self.imgsz[0], max_det=self.max_det,
                                   verbose=False)
            return [np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                     r.boxes.cls.cpu().numpy()]).astype(np.float32).reshape(-1, 6) for r in res]
        if self._sess is not None and getattr(self, "dynamic_batch", False) and len(frames) > 1:
            metas, blobs = [], []
            for f in frames:
                img, r, pad = letterbox(f, self.imgsz)
                blobs.append(img)
                metas.append((r, pad, f.shape[1], f.shape[0]))
            out = self._forward(cv2.dnn.blobFromImages(blobs, 1 / 255.0, swapRB=True))
            return [self._postprocess(out[i:i + 1], conf, *m) for i, m in enumerate(metas)]
        return [self._detect_array(f, conf) for f in frames]

    def _detect_array(self, bgr, conf):
        h, w = bgr.shape[:2]
        img, r, pad = letterbox(bgr, self.imgsz)
        return self._postprocess(self._forward(cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)), conf, r, pad, w, h)

    def _forward(self, blob):
        if self._sess is not None:
            return self._sess.run(None, {self._in_name: blob.astype(np.float16) if self._fp16 else blob})[0]
        self._net.setInput(blob)
        return self._net.forward()

    def detect_files(self, paths, conf=None, batch=8):
        out = []
        for i in range(0, len(paths), batch):
            frames = [imread(p) for p in paths[i:i + batch]]
            ok = [f for f in frames if f is not None]
            res = iter(self.detect_batch(ok, conf) if ok else [])
            out += [next(res) if f is not None else None for f in frames]
        return out

    def detect(self, bgr, conf=None):
        conf = self.conf if conf is None else conf
        h, w = bgr.shape[:2]
        if self._ul is not None:
            r = self._ul.predict(bgr, conf=conf, iou=self.iou, imgsz=self.imgsz[0], verbose=False)[0]
            b = r.boxes
            return [Detection(*map(float, xy), float(c), int(k), self.name_of(k), self.label)
                    for xy, c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]
        return [Detection(*map(float, row[:5]), int(row[5]), self.name_of(row[5]), self.label)
                for row in self._detect_array(bgr, conf)]


# ------------------------------------------------------ environment models
_CLIENTS = {}          # python executable -> InferenceClient (one server per env per process)
_CLIENT_LOCK = None


def _client(python):
    import threading, atexit
    global _CLIENT_LOCK
    if _CLIENT_LOCK is None:
        _CLIENT_LOCK = threading.Lock()
    with _CLIENT_LOCK:
        c = _CLIENTS.get(python)
        if c is None or not c.alive:
            from AIO_QC.tools.frame_extractor.core.inference import InferenceClient
            c = InferenceClient(python)
            c.start()
            c.info = c.probe()
            _CLIENTS[python] = c
            if len(_CLIENTS) == 1:
                atexit.register(shutdown_clients)
        return c


def shutdown_clients():
    for c in list(_CLIENTS.values()):
        try:
            c.close(timeout=3)
        except Exception:
            pass
    _CLIENTS.clear()


class _FrameFile:
    """The frame handed to the server as .npy; written once per image even for several models."""
    _last = {}

    @classmethod
    def path_for(cls, img):
        import tempfile, numpy as _np
        key = (img.__array_interface__["data"][0], img.shape, img.strides)
        path = os.path.join(tempfile.gettempdir(), f"fx_infer_{os.getpid()}.npy")
        if cls._last.get(path) != key:
            _np.save(path, _np.ascontiguousarray(img))
            cls._last[path] = key
        return path


class RemoteDetector:
    """.pt / .engine model served by infer_server in another Python environment."""
    batch = 8

    def __init__(self, path, python, conf=0.25, iou=0.45, imgsz=0, device="cpu", half=False, sahi=None,
                 max_det=300):
        import hashlib
        self.path, self.python = path, python
        self.label = os.path.splitext(os.path.basename(path))[0]
        self.conf, self.iou = conf, iou
        dev = "cpu" if device in ("auto", "", None) else str(device)
        self.client = _client(python)
        self.provider = f"env · {dev}"
        spec = {"id": "", "path": os.path.abspath(path), "imgsz": int(imgsz) or None, "device": dev,
                "half": bool(half) and dev not in ("cpu", "mps"), "iou": iou, "warmup": True, "conf": 0.05,
                "sahi": sahi, "max_det": int(max_det)}
        spec["id"] = hashlib.sha1(repr(sorted(spec.items())).encode()).hexdigest()[:12]
        self.id = spec["id"]
        info = self.client.load([spec])[0]
        if not info.get("ok"):
            raise RuntimeError(f"{os.path.basename(path)}: {info.get('error')}")
        self.names = {int(k): v for k, v in info.get("names", {}).items()}
        self.task = info.get("task", "detect")
        sz = spec["imgsz"] or info.get("train_imgsz") or 640
        self.imgsz = (sz, sz)

    def name_of(self, c):
        return self.names.get(int(c), str(int(c)))

    def detect_batch(self, frames, conf=None, cancel=None):
        """In-memory frames (shared memory, batched on the server) -> list of (n, 6) float32 arrays."""
        conf = self.conf if conf is None else conf
        res = self.client.predict_frames(frames, [self.id], conf=conf, batch=self.batch, cancel=cancel)
        out = []
        for r in res:
            if r.get("error") and r["by_model"].get(self.id) is None:
                raise RuntimeError(r["error"])
            out.append(_rows(r["by_model"].get(self.id)))
        return out

    def detect_files(self, paths, conf=None, batch=None, cancel=None):
        """Image files read by the server itself (no transfer at all), batched. None for unreadable files."""
        conf = self.conf if conf is None else conf
        res = self.client.predict(paths, [self.id], conf=conf, batch=batch or self.batch, cancel=cancel)
        out = []
        for p in paths:
            r = res.get(str(p)) or {}
            boxes = (r.get("by_model") or {}).get(self.id)
            out.append(None if boxes is None and r.get("error") else _rows(boxes))
        return out

    def detect(self, bgr, conf=None):
        conf = self.conf if conf is None else conf
        p = _FrameFile.path_for(bgr)
        res = self.client.predict_arrays([p], [self.id], conf=conf)[p]
        if res.get("error") and not res["by_model"].get(self.id):
            raise RuntimeError(res["error"])
        return [Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), int(b[5]),
                          self.name_of(b[5]), self.label) for b in (res["by_model"].get(self.id) or [])]


def model_backend(path, env_python=""):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        return "onnxruntime" if HAVE_ORT else "cv2.dnn"
    if env_python:
        return "environment"
    try:
        import importlib.util
        if importlib.util.find_spec("ultralytics"):
            return "ultralytics (in-process)"
    except Exception:
        pass
    return "needs environment"


def load_detector(path, conf=0.25, iou=0.45, imgsz=0, device="auto", threads=0, env_python="", half=False,
                  sahi=None, max_det=300):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".engine", ".torchscript", ".mlpackage") or (ext != ".onnx" and env_python):
        if env_python:
            return RemoteDetector(path, env_python, conf, iou, imgsz, device, half, sahi, max_det)
    d = YoloDetector(path, conf, iou, imgsz, "cpu" if device == "cpu" else "auto", threads)
    d.max_det = int(max_det)
    return d


def load_detectors(paths, conf=0.25, iou=0.45, imgsz=0, device="auto", threads=0, env_python="", half=False,
                   sahi=None):
    return [load_detector(p, conf, iou, imgsz, device, threads, env_python, half, sahi) for p in paths]


# ------------------------------------------------------------------ mining
MINING_CRITERIA = {
    "uncertain": "Any detection with confidence inside the uncertainty band",
    "empty": "No detections at all (hard negatives / missed objects)",
    "disagree": "Models disagree (unmatched box of a class both models know)",
    "classes": "Contains one of the chosen (e.g. rare) classes",
    "temporal": "Temporal inconsistency: a tracked object vanishes (flicker) or appears only briefly (blip)",
}


def mining_reasons(per_model, criteria, conf, unc_low, unc_high, classes=(), match_iou=0.5, models=()):
    """per_model: list of detection lists (one per model, detected at conf=min(conf, unc_low)).
    Returns the list of criteria this frame satisfies."""
    reasons = []
    confident = [[d for d in dets if d.conf >= conf] for dets in per_model]
    if "uncertain" in criteria and any(unc_low <= d.conf < unc_high for dets in per_model for d in dets):
        reasons.append("uncertain")
    if "empty" in criteria and not any(confident):
        reasons.append("empty")
    if "classes" in criteria and classes:
        wanted = {c.strip().lower() for c in classes}
        if any(d.name.lower() in wanted or str(d.cls) in wanted for dets in confident for d in dets):
            reasons.append("classes")
    if "disagree" in criteria and len(per_model) >= 2:
        names = [set(m.names.values()) if getattr(m, "names", None) else None for m in models] \
            or [None] * len(per_model)
        done = False
        for i in range(len(confident)):
            for j in range(i + 1, len(confident)):
                common = (names[i] & names[j]) if names[i] and names[j] else None
                a = [d for d in confident[i] if common is None or d.name in common]
                b = [d for d in confident[j] if common is None or d.name in common]
                if _unmatched(a, b, match_iou) or _unmatched(b, a, match_iou):
                    reasons.append("disagree")
                    done = True
                    break
            if done:
                break
    return reasons


def _unmatched(a, b, thr):
    for d in a:
        if not any(e.name == d.name and d.iou(e) >= thr for e in b):
            return True
    return False


# ------------------------------------------------------- pre-annotations
def write_labelme(json_path, image_name, w, h, dets, flags=None):
    import json
    shapes = [{
        "label": d.name,
        "points": [[round(d.x1, 2), round(d.y1, 2)], [round(d.x2, 2), round(d.y2, 2)]],
        "group_id": None,
        "description": f"conf={d.conf:.3f} model={d.model}",
        "shape_type": "rectangle",
        "flags": {},
    } for d in dets]
    data = {"version": "5.2.1", "flags": flags or {}, "shapes": shapes, "imagePath": image_name,
            "imageData": None, "imageHeight": int(h), "imageWidth": int(w)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_yolo(txt_path, w, h, dets):
    with open(txt_path, "w", encoding="utf-8") as f:
        for d in dets:
            cx, cy = (d.x1 + d.x2) / 2 / w, (d.y1 + d.y2) / 2 / h
            bw, bh = (d.x2 - d.x1) / w, (d.y2 - d.y1) / h
            f.write(f"{d.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
