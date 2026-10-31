# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
AIO QC inference server — runs INSIDE the user's chosen Python environment.

Standalone on purpose: standard library + ultralytics (+ optional sahi). The app starts it as
    <env python> -u infer_server.py
and talks JSON over stdin/stdout, one request per line:
    {"rid": 1, "cmd": "probe"}
    {"rid": 2, "cmd": "load", "models": [{"id", "path", "imgsz", "device", "half", "sahi": null | {...}}]}
    {"rid": 3, "cmd": "predict", "images": [...], "ids": [...], "conf": 0.05, "batch": 8, "classes": {id: [..]}}
    {"rid": 4, "cmd": "shutdown"}
Responses are single lines prefixed with RESP_PREFIX on the ORIGINAL stdout. Everything libraries print
goes to stderr (sys.stdout is redirected), which the app shows in its console.
"""
import json
import os
import platform
import sys
import time
import traceback

RESP_PREFIX = "@@AIOQC@@ "
_OUT = sys.stdout
sys.stdout = sys.stderr            # keep library chatter away from the protocol channel

MODELS = {}                        # id -> {"kind": "ultralytics"|"sahi", "model", "spec", "names"}


def reply(rid, **payload):
    payload["rid"] = rid
    _OUT.write(RESP_PREFIX + json.dumps(payload) + "\n")
    _OUT.flush()


def log(msg):
    sys.stderr.write(f"[server] {msg}\n")
    sys.stderr.flush()


def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def probe():
    info = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
            "ultralytics": _version("ultralytics"), "torch": None, "torchvision": _version("torchvision"),
            "sahi": _version("sahi"), "onnxruntime": _version("onnxruntime"), "tensorrt": _version("tensorrt"),
            "cuda": False, "cuda_version": None, "mps": False,
            "devices": [{"id": "cpu", "name": "CPU"}]}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({"id": str(i), "name": f"cuda:{i} · {p.name} · {p.total_memory / 2**30:.1f} GB"})
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            info["mps"] = True
            info["devices"].append({"id": "mps", "name": "Apple MPS"})
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _train_imgsz(model):
    for src in (getattr(model, "overrides", None) or {}, (getattr(model, "ckpt", None) or {}).get("train_args", {})):
        v = src.get("imgsz") if isinstance(src, dict) else None
        if isinstance(v, (list, tuple)):
            v = max(v)
        if v:
            return int(v)
    return None


def load(specs):
    out = []
    keep = {s["id"] for s in specs}
    for mid in list(MODELS):
        if mid not in keep:
            MODELS.pop(mid, None)
    for spec in specs:
        mid = spec["id"]
        old = MODELS.get(mid)
        if old is not None and old["spec"] == spec:
            out.append(old["info"])
            continue
        t0 = time.time()
        try:
            from ultralytics import YOLO
            y = YOLO(spec["path"])
            names = {str(k): str(v) for k, v in dict(y.names).items()}
            info = {"id": mid, "ok": True, "names": names, "task": getattr(y, "task", "detect"),
                    "train_imgsz": _train_imgsz(y)}
            entry = {"kind": "ultralytics", "model": y, "spec": spec, "names": names, "info": info}
            if spec.get("sahi"):
                entry["kind"] = "sahi"
                entry["sahi_model"] = _load_sahi(spec)
            elif spec.get("warmup", True):
                try:   # first call compiles / allocates: pay it now, not on the first image
                    import numpy as np
                    y.predict(np.zeros((64, 64, 3), dtype="uint8"), imgsz=spec.get("imgsz") or 640,
                              device=spec.get("device") or None, half=bool(spec.get("half")), verbose=False)
                except Exception as exc:
                    log(f"warm-up skipped for {spec['path']}: {exc}")
            MODELS[mid] = entry
            info["load_s"] = round(time.time() - t0, 2)
            log(f"loaded {os.path.basename(spec['path'])} ({info['task']}, {len(names)} classes) in {info['load_s']}s")
            out.append(info)
        except Exception as exc:
            log(traceback.format_exc())
            out.append({"id": mid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return out


def _load_sahi(spec):
    from sahi import AutoDetectionModel
    last = None
    for model_type in ("ultralytics", "yolov8"):              # newer / older sahi names
        try:
            return AutoDetectionModel.from_pretrained(
                model_type=model_type, model_path=spec["path"], confidence_threshold=spec.get("conf", 0.05),
                device=("cuda:" + spec["device"]) if str(spec.get("device", "")).isdigit() else (spec.get("device") or "cpu"),
                image_size=spec.get("imgsz") or None)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"SAHI could not load the model: {last}")


def _boxes_from_result(r, keep):
    boxes = []
    src = r.boxes if (getattr(r, "boxes", None) is not None and len(r.boxes)) else getattr(r, "obb", None)
    if src is None or not len(src):
        return boxes
    xyxy = src.xyxy.cpu().numpy().tolist()
    conf = src.conf.cpu().numpy().tolist()
    cls = src.cls.cpu().numpy().astype(int).tolist()
    for b, c, k in zip(xyxy, conf, cls):
        if keep is None or k in keep:
            boxes.append([round(float(v), 2) for v in b] + [round(float(c), 4), int(k)])
    return boxes


def predict(images, ids, conf, batch, classes):
    results = {p: {"w": None, "h": None, "by_model": {}, "error": None} for p in images}
    for mid in ids:
        entry = MODELS.get(mid)
        if entry is None:
            for p in images:
                results[p]["by_model"][mid] = None
            continue
        spec = entry["spec"]
        keep = set(classes.get(mid)) if classes and classes.get(mid) is not None else None
        if entry["kind"] == "sahi":
            from sahi.predict import get_sliced_prediction
            s = spec["sahi"]
            for p in images:
                try:
                    r = get_sliced_prediction(p, entry["sahi_model"], slice_height=s.get("slice", 640),
                                              slice_width=s.get("slice", 640),
                                              overlap_height_ratio=s.get("overlap", 0.2),
                                              overlap_width_ratio=s.get("overlap", 0.2), verbose=0)
                    boxes = []
                    for op in r.object_prediction_list:
                        k = int(op.category.id)
                        if op.score.value >= conf and (keep is None or k in keep):
                            b = op.bbox
                            boxes.append([round(b.minx, 2), round(b.miny, 2), round(b.maxx, 2), round(b.maxy, 2),
                                          round(float(op.score.value), 4), k])
                    results[p]["by_model"][mid] = boxes
                    results[p]["w"], results[p]["h"] = r.image_width, r.image_height
                except Exception as exc:
                    results[p]["error"] = f"{mid}: {type(exc).__name__}: {exc}"
            continue
        kw = dict(imgsz=spec.get("imgsz") or 640, conf=conf, device=spec.get("device") or None,
                  half=bool(spec.get("half")), verbose=False)
        if keep is not None:
            kw["classes"] = sorted(keep)           # skip unused classes inside NMS
        for i in range(0, len(images), max(1, batch)):
            chunk = images[i:i + max(1, batch)]
            try:
                try:
                    res = entry["model"].predict(chunk, batch=len(chunk), **kw)
                except TypeError:                                  # older ultralytics: no batch kwarg
                    res = entry["model"].predict(chunk, **kw)
                for p, r in zip(chunk, res):
                    h, w = r.orig_shape[:2]
                    results[p]["w"], results[p]["h"] = int(w), int(h)
                    results[p]["by_model"][mid] = _boxes_from_result(r, keep)
            except Exception as exc:                               # retry one by one to isolate a bad image
                for p in chunk:
                    try:
                        r = entry["model"].predict(p, **kw)[0]
                        h, w = r.orig_shape[:2]
                        results[p]["w"], results[p]["h"] = int(w), int(h)
                        results[p]["by_model"][mid] = _boxes_from_result(r, keep)
                    except Exception as exc2:
                        results[p]["error"] = f"{mid}: {type(exc2).__name__}: {exc2}"
                log(f"batch fallback ({type(exc).__name__}: {exc})")
    return results


def export(spec):
    """Export with ultralytics (it installs onnx / tensorrt requirements itself when online)."""
    from ultralytics import YOLO
    t0 = time.time()
    fmt = spec["format"]
    kw = dict(format=fmt, imgsz=int(spec.get("imgsz") or 640), half=bool(spec.get("half")),
              device=spec.get("device") or None)
    if fmt == "onnx":
        kw.update(dynamic=True, simplify=True)
    elif fmt == "engine":
        kw.update(dynamic=True, batch=int(spec.get("batch") or 8), workspace=spec.get("workspace") or None)
    log(f"exporting {os.path.basename(spec['path'])} → {fmt} ({kw})")
    out = YOLO(spec["path"]).export(**{k: v for k, v in kw.items() if v is not None})
    log(f"export finished in {time.time() - t0:.1f}s: {out}")
    return {"path": str(out), "seconds": round(time.time() - t0, 1)}


def main():
    log(f"ready · python {sys.version.split()[0]} · {sys.executable}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("rid")
            cmd = req.get("cmd")
            if cmd == "probe":
                reply(rid, ok=True, info=probe())
            elif cmd == "load":
                reply(rid, ok=True, models=load(req["models"]))
            elif cmd == "predict":
                reply(rid, ok=True, results=predict(req["images"], req["ids"], float(req.get("conf", 0.05)),
                                                    int(req.get("batch", 8)), req.get("classes") or {}))
            elif cmd == "export":
                reply(rid, ok=True, **export(req["spec"]))
            elif cmd == "shutdown":
                reply(rid, ok=True)
                break
            else:
                reply(rid, ok=False, error=f"unknown command {cmd!r}")
        except Exception as exc:
            log(traceback.format_exc())
            reply(rid, ok=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
