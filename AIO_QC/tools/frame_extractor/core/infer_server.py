# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
FrameExtractor inference server — runs INSIDE the user's chosen Python environment.
(Adapted from AIO QC's infer_server: same protocol, plus video-frame input and additive loading.)

Standalone on purpose: standard library + ultralytics (+ optional sahi). The app starts it as
    <env python> -u infer_server.py
and talks JSON over stdin/stdout, one request per line:
    {"rid": 1, "cmd": "probe"}
    {"rid": 2, "cmd": "load", "models": [{"id", "path", "imgsz", "device", "half", "sahi": null | {...}}]}
    {"rid": 3, "cmd": "predict", "images": [...], "ids": [...], "conf": 0.05, "batch": 8, "classes": {id: [..]}}
    {"rid": 4, "cmd": "predict_arrays", "arrays": ["frame.npy", ...], "ids": [...], "conf": 0.05}
         BGR uint8 frames saved with numpy.save by the app (no JPEG round-trip, exact pixels)
    {"rid": 5, "cmd": "predict_shm", "shm": "<name>", "frames": [[offset, h, w], ...], "ids": [...], "conf": ..}
         BGR uint8 frames in a shared-memory block owned by the app: zero-copy, no disk, batched
    {"rid": 6, "cmd": "embed_load", "spec": {"backend": "auto"|"open_clip"|"transformers", "model", "pretrained",
                                             "device"}}            CLIP-style image / text embeddings (frame search)
    {"rid": 7, "cmd": "embed_shm", "shm": .., "frames": [[offset, h, w], ...]}   -> unit vectors
    {"rid": 8, "cmd": "embed_text", "texts": [...]}                            -> unit vectors
    {"rid": 4, "cmd": "shutdown"}
Responses are single lines prefixed with RESP_PREFIX on the ORIGINAL stdout. Everything libraries print
goes to stderr (sys.stdout is redirected), which the app shows in its console.
"""
import os as _os
import sys as _sys

# Started as `python <this file>`, Python puts this folder first on sys.path - and the app's own modules here
# (profile.py, io.py, imageio.py …) would shadow the standard library / packages the models need:
# torch imports cProfile, which imports `profile`, and got AIO QC's instead ("cannot import name 'YOLO' from
# 'ultralytics'"). This server needs nothing from its folder: take it off the path before anything else is imported.
_HERE = _os.path.normcase(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:] = [p for p in _sys.path if _os.path.normcase(_os.path.abspath(p or _os.curdir)) != _HERE]
del _HERE

import json
import os
import platform
import sys
import time
import traceback

RESP_PREFIX = "@@FXINFER@@ "
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


def load(specs, replace=False):
    out = []
    if replace:
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
            for p in images:
                try:
                    boxes, (w, h) = _sahi_predict(p, entry, conf, keep)
                    results[p]["by_model"][mid] = boxes
                    results[p]["w"], results[p]["h"] = w, h
                except Exception as exc:
                    results[p]["error"] = f"{mid}: {type(exc).__name__}: {exc}"
            continue
        kw = _kw(spec, conf, keep)
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


def predict_arrays(paths, ids, conf, batch, classes):
    """Same as predict() but the inputs are .npy BGR frames (keys of the result are the .npy paths)."""
    import numpy as np
    arrays = {}
    for p in paths:
        try:
            arrays[p] = np.load(p)
        except Exception as exc:
            arrays[p] = exc
    results = {p: {"w": None, "h": None, "by_model": {}, "error": None} for p in paths}
    good = [p for p in paths if not isinstance(arrays[p], Exception)]
    for p in paths:
        if isinstance(arrays[p], Exception):
            results[p]["error"] = f"load: {arrays[p]}"
    for mid in ids:
        entry = MODELS.get(mid)
        if entry is None:
            continue
        spec = entry["spec"]
        keep = set(classes.get(mid)) if classes and classes.get(mid) is not None else None
        pending = []
        for p in good:
            img = arrays[p]
            h, w = img.shape[:2]
            results[p]["w"], results[p]["h"] = int(w), int(h)
            try:
                if entry["kind"] == "sahi":
                    results[p]["by_model"][mid] = _sahi_boxes(img, entry, conf, keep)
                    continue
                pending.append(p)
            except Exception as exc:
                results[p]["error"] = f"{mid}: {type(exc).__name__}: {exc}"
        if entry["kind"] != "sahi" and pending:
            _predict_batch([arrays[p] for p in pending], pending, entry, conf, keep, batch, results, mid)
    return results


def _kw(spec, conf, keep):
    kw = dict(imgsz=spec.get("imgsz") or 640, conf=conf, device=spec.get("device") or None,
              half=bool(spec.get("half")), verbose=False)
    if spec.get("iou"):
        kw["iou"] = float(spec["iou"])
    if spec.get("max_det"):
        kw["max_det"] = int(spec["max_det"])
    if keep is not None:
        kw["classes"] = sorted(keep)           # skip unused classes inside NMS
    return kw


def _predict_batch(imgs, keys, entry, conf, keep, batch, results, mid):
    """Batched ultralytics predict on in-memory BGR frames; isolates a failing frame by retrying singly."""
    kw = _kw(entry["spec"], conf, keep)
    step = max(1, int(batch))
    for i in range(0, len(imgs), step):
        chunk, ck = imgs[i:i + step], keys[i:i + step]
        try:
            try:
                res = entry["model"].predict(list(chunk), batch=len(chunk), **kw)
            except TypeError:
                res = entry["model"].predict(list(chunk), **kw)
            for k, r in zip(ck, res):
                results[k]["by_model"][mid] = _boxes_from_result(r, keep)
        except Exception as exc:
            log(f"batch fallback ({type(exc).__name__}: {exc})")
            for img, k in zip(chunk, ck):
                try:
                    results[k]["by_model"][mid] = _boxes_from_result(entry["model"].predict(img, **kw)[0], keep)
                except Exception as exc2:
                    results[k]["error"] = f"{mid}: {type(exc2).__name__}: {exc2}"


def predict_shm(name, frames, ids, conf, batch, classes):
    """Frames in a shared-memory block created by the app (it owns and frees it)."""
    import numpy as np
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=name, create=False, track=False)      # Python 3.13+
    except TypeError:
        shm = shared_memory.SharedMemory(name=name, create=False)
        try:   # older Pythons register attached blocks and would unlink the app's block at our exit
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    try:
        imgs = [np.ndarray((int(h), int(w), 3), np.uint8, buffer=shm.buf, offset=int(off))
                for off, h, w in frames]
        keys = [str(i) for i in range(len(imgs))]
        results = {k: {"w": int(f[2]), "h": int(f[1]), "by_model": {}, "error": None} for k, f in zip(keys, frames)}
        for mid in ids:
            entry = MODELS.get(mid)
            if entry is None:
                continue
            keep = set(classes.get(mid)) if classes and classes.get(mid) is not None else None
            if entry["kind"] == "sahi":
                for k, img in zip(keys, imgs):
                    try:
                        results[k]["by_model"][mid] = _sahi_boxes(np.ascontiguousarray(img), entry, conf, keep)
                    except Exception as exc:
                        results[k]["error"] = f"{mid}: {type(exc).__name__}: {exc}"
                continue
            _predict_batch(imgs, keys, entry, conf, keep, batch, results, mid)
        del imgs
        return results
    finally:
        shm.close()


def _sahi_predict(image, entry, conf, keep):
    """SAHI sliced prediction on a file path or an RGB array -> (boxes [x1,y1,x2,y2,score,cls], (w, h))."""
    from sahi.predict import get_sliced_prediction
    s = entry["spec"]["sahi"]
    r = get_sliced_prediction(image, entry["sahi_model"], slice_height=s.get("slice", 640),
                              slice_width=s.get("slice", 640), overlap_height_ratio=s.get("overlap", 0.2),
                              overlap_width_ratio=s.get("overlap", 0.2), verbose=0)
    boxes = []
    for op in r.object_prediction_list:
        k = int(op.category.id)
        if op.score.value >= conf and (keep is None or k in keep):
            b = op.bbox
            boxes.append([round(b.minx, 2), round(b.miny, 2), round(b.maxx, 2), round(b.maxy, 2),
                          round(float(op.score.value), 4), k])
    return boxes, (r.image_width, r.image_height)


def _sahi_boxes(img_bgr, entry, conf, keep):
    return _sahi_predict(img_bgr[:, :, ::-1].copy(), entry, conf, keep)[0]


EMBED = {}      # the loaded embedding model (one at a time)


def _attach(name):
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=name, create=False)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    return shm


def embed_load(spec):
    """Load a CLIP model: open_clip if installed, else Hugging Face transformers."""
    key = repr(sorted(spec.items()))
    if EMBED.get("key") == key:
        return EMBED["info"]
    import torch
    dev = spec.get("device") or "cpu"
    dev = f"cuda:{dev}" if str(dev).isdigit() else ("cpu" if dev in ("auto", "") and not torch.cuda.is_available()
                                                     else ("cuda" if dev in ("auto", "") else dev))
    backend = spec.get("backend") or "auto"
    errors = []
    if backend in ("auto", "open_clip"):
        try:
            import open_clip
            name = spec.get("model") or "ViT-B-32"
            pre = spec.get("pretrained") or "laion2b_s34b_b79k"
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
            model.eval()
            tok = open_clip.get_tokenizer(name)
            EMBED.clear()
            EMBED.update(key=key, kind="open_clip", model=model, pre=preprocess, tok=tok, dev=dev)
            EMBED["info"] = dict(backend="open_clip", model=f"{name}/{pre}", device=dev)
            return EMBED["info"]
        except ImportError as exc:
            errors.append(f"open_clip: {exc}")
    if backend in ("auto", "transformers"):
        try:
            from transformers import CLIPModel, CLIPProcessor
            mid = spec.get("hf_model") or "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(mid).to(dev).eval()
            proc = CLIPProcessor.from_pretrained(mid)
            EMBED.clear()
            EMBED.update(key=key, kind="transformers", model=model, proc=proc, dev=dev)
            EMBED["info"] = dict(backend="transformers", model=mid, device=dev)
            return EMBED["info"]
        except ImportError as exc:
            errors.append(f"transformers: {exc}")
    raise RuntimeError("No CLIP library in this environment - install one: pip install open_clip_torch  "
                       "(or: pip install transformers). " + " | ".join(errors))


def _unit(x):
    import numpy as np
    x = np.asarray(x, dtype="float32")
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).round(5).tolist()


def embed_images(imgs_bgr):
    import torch
    from PIL import Image
    e = EMBED
    pil = [Image.fromarray(im[:, :, ::-1].copy()) for im in imgs_bgr]
    with torch.no_grad():
        if e["kind"] == "open_clip":
            x = torch.stack([e["pre"](p) for p in pil]).to(e["dev"])
            v = e["model"].encode_image(x)
        else:
            inp = e["proc"](images=pil, return_tensors="pt").to(e["dev"])
            v = e["model"].get_image_features(**inp)
    return _unit(v.float().cpu().numpy())


def embed_text(texts):
    import torch
    e = EMBED
    with torch.no_grad():
        if e["kind"] == "open_clip":
            v = e["model"].encode_text(e["tok"](texts).to(e["dev"]))
        else:
            inp = e["proc"](text=texts, return_tensors="pt", padding=True).to(e["dev"])
            v = e["model"].get_text_features(**inp)
    return _unit(v.float().cpu().numpy())


def embed_shm(name, frames):
    import numpy as np
    if not EMBED:
        raise RuntimeError("no embedding model loaded")
    shm = _attach(name)
    try:
        imgs = [np.ndarray((int(h), int(w), 3), np.uint8, buffer=shm.buf, offset=int(off)).copy() for off, h, w in frames]
    finally:
        shm.close()
    return embed_images(imgs)


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
                reply(rid, ok=True, models=load(req["models"], bool(req.get("replace"))))
            elif cmd == "predict_arrays":
                reply(rid, ok=True, results=predict_arrays(req["arrays"], req["ids"], float(req.get("conf", 0.05)),
                                                           int(req.get("batch", 8)), req.get("classes") or {}))
            elif cmd == "predict_shm":
                reply(rid, ok=True, results=predict_shm(req["shm"], req["frames"], req["ids"],
                                                        float(req.get("conf", 0.05)), int(req.get("batch", 8)),
                                                        req.get("classes") or {}))
            elif cmd == "embed_load":
                reply(rid, ok=True, info=embed_load(req["spec"]))
            elif cmd == "embed_shm":
                reply(rid, ok=True, vectors=embed_shm(req["shm"], req["frames"]))
            elif cmd == "embed_text":
                reply(rid, ok=True, vectors=embed_text(req["texts"]))
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
