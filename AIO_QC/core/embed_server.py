# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC embedding server — runs INSIDE the user's Python environment (like infer_server.py / ocr_server.py).

Turns image crops (or whole images) into vectors with a vision model the user already has:
    DINOv3 / DINOv2 · SigLIP / SigLIP2 · CLIP — any Hugging Face transformers folder, a timm folder, an open_clip
    folder, or the original DINOv3 repo + .pth weights (torch.hub, local)
The app never imports torch. Protocol: one JSON request per line on stdin, replies prefixed with RESP_PREFIX:
    {"cmd": "probe"}
    {"cmd": "download", "repo": "facebook/dinov3-vits16-pretrain-lvd1689m", "dest": "...", "token": "hf_…"}
    {"cmd": "load", "path": "...", "device": "auto|cpu|cuda:0|mps", "half": true}
    {"cmd": "embed", "items": [{"image": path, "boxes": [[x1,y1,x2,y2] | null, …]}], "out": "x.npy",
                     "batch": 32, "pad": 0.1}           -> float16 [n, dim], L2-normalised, in item / box order
    {"cmd": "knn", "inp": "x.npy", "k": 10, "out": "nn"} -> nn_idx.npy (int32), nn_sim.npy (float16)
Large arrays travel as .npy files, never through the pipe.
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
import sys
import traceback

RESP_PREFIX = "@@AIOQC@@ "
_OUT = sys.stdout
sys.stdout = sys.stderr

STATE = {"model": None, "kind": None, "pre": None, "device": "cpu", "half": False, "dim": None, "path": None,
         "size": 224, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}


def reply(rid, **payload):
    payload["rid"] = rid
    _OUT.write(RESP_PREFIX + json.dumps(payload) + "\n")
    _OUT.flush()


def log(msg):
    sys.stderr.write(f"[embed] {msg}\n")
    sys.stderr.flush()


def _ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def probe():
    info = {"python": sys.version.split()[0], "torch": _ver("torch"), "transformers": _ver("transformers"),
            "timm": _ver("timm"), "open_clip": _ver("open_clip"), "huggingface_hub": _ver("huggingface_hub"),
            "faiss": _ver("faiss"), "PIL": _ver("PIL"), "devices": [{"id": "cpu", "name": "CPU"}]}
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({"id": f"cuda:{i}", "name": f"cuda:{i} · {p.name} · {p.total_memory / 2**30:.1f} GB"})
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            info["devices"].append({"id": "mps", "name": "Apple MPS"})
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


# ---------------------------------------------------------------------- model kinds
def detect(path):
    """(kind, root) for a model folder: torchhub | timm | open_clip | transformers."""
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        fs = set(files)
        if "hubconf.py" in fs:
            return "torchhub", root
        if "open_clip_config.json" in fs:
            return "open_clip", root
        if "config.json" in fs:
            try:
                cfg = json.load(open(os.path.join(root, "config.json"), encoding="utf-8"))
            except Exception:
                cfg = {}
            if "architecture" in cfg and "model_type" not in cfg:
                return "timm", root
            return "transformers", root
    raise ValueError("no model found: expected a Hugging Face folder (config.json), a timm / open_clip folder, "
                     "or the DINOv3 repository (hubconf.py) with a .pth weights file")


def _device(req):
    import torch
    if req in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


def load(path, device="auto", half=True):
    import torch
    kind, root = detect(path)
    dev = _device(device)
    half = bool(half) and dev.startswith("cuda")
    model = pre = None
    size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if kind == "transformers":
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(root, torch_dtype=torch.float16 if half else torch.float32)
        try:
            pre = AutoImageProcessor.from_pretrained(root)
            s = getattr(pre, "size", None) or {}
            size = int(s.get("height") or s.get("shortest_edge") or s.get("width") or 224) if isinstance(s, dict) \
                else int(s)
            mean = tuple(getattr(pre, "image_mean", None) or mean)
            std = tuple(getattr(pre, "image_std", None) or std)
        except Exception as exc:
            log(f"no image processor ({exc}); using 224 px, ImageNet normalisation")
    elif kind == "timm":
        import timm
        cfg = json.load(open(os.path.join(root, "config.json"), encoding="utf-8"))
        weights = next((os.path.join(root, f) for f in ("model.safetensors", "pytorch_model.bin")
                        if os.path.isfile(os.path.join(root, f))), None)
        model = timm.create_model(cfg["architecture"], pretrained=weights is not None, num_classes=0,
                                  pretrained_cfg_overlay=dict(file=weights) if weights else None)
        dc = timm.data.resolve_data_config(model=model)
        size, mean, std = int(dc["input_size"][-1]), tuple(dc["mean"]), tuple(dc["std"])
    elif kind == "open_clip":
        import open_clip
        cfg = json.load(open(os.path.join(root, "open_clip_config.json"), encoding="utf-8"))
        weights = next((os.path.join(root, f) for f in ("open_clip_pytorch_model.bin", "open_clip_model.safetensors")
                        if os.path.isfile(os.path.join(root, f))), None)
        name = cfg.get("model_cfg", {}).get("name") or cfg.get("model_name") or "ViT-B-32"
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=weights)
        pc = cfg.get("preprocess_cfg", {})
        size = int((pc.get("size") or [224])[0] if isinstance(pc.get("size"), list) else pc.get("size") or 224)
        mean, std = tuple(pc.get("mean") or (0.481, 0.458, 0.408)), tuple(pc.get("std") or (0.269, 0.261, 0.276))
    else:                                                            # DINOv3 repo + .pth
        pth = None
        for r, _d, files in os.walk(path):
            for f in files:
                if f.endswith(".pth"):
                    pth = os.path.join(r, f)
        if pth is None:
            raise ValueError("the DINOv3 repository needs a .pth weights file next to it")
        name = os.path.basename(pth).lower()
        arch = next((a for a in ("vit7b16", "vith16plus", "vitl16plus", "vitl16", "vitb16", "vits16plus", "vits16",
                                 "convnext_large", "convnext_base", "convnext_small", "convnext_tiny")
                     if a in name.replace("_", "").replace("-", "") or a in name), "vits16")
        model = torch.hub.load(root, f"dinov3_{arch}", source="local", weights=pth)
    model.eval().to(dev)
    if half and kind != "transformers":
        model.half()
    STATE.update(model=model, kind=kind, pre=pre, device=dev, half=half, path=path, size=size, mean=mean, std=std)
    with torch.no_grad():
        import numpy as np
        dummy = _forward([np.zeros((size, size, 3), np.uint8)])
    STATE["dim"] = int(dummy.shape[1])
    return {"kind": kind, "root": root, "device": dev, "half": half, "dim": STATE["dim"], "size": size}


def _to_tensor(arrs):
    import numpy as np
    import torch
    s = STATE["size"]
    out = np.stack([a for a in arrs]).astype(np.float32) / 255.0
    out = (out - np.array(STATE["mean"], np.float32)) / np.array(STATE["std"], np.float32)
    t = torch.from_numpy(out.transpose(0, 3, 1, 2).copy())
    assert t.shape[-1] == s
    return t.to(STATE["device"], dtype=torch.float16 if STATE["half"] else torch.float32)


def _forward(arrs):
    """uint8 HxWx3 squares (already STATE['size']) -> L2-normalised float32 numpy [n, dim]."""
    import torch
    model, kind = STATE["model"], STATE["kind"]
    with torch.no_grad():
        if kind == "transformers":
            pre = STATE["pre"]
            if pre is not None:                                       # the model's own preprocessing (SigLIP2 …)
                from PIL import Image
                inputs = pre(images=[Image.fromarray(a) for a in arrs], return_tensors="pt")
                inputs = {k: (v.to(STATE["device"], dtype=torch.float16 if STATE["half"] else torch.float32)
                              if v.is_floating_point() else v.to(STATE["device"]))
                          for k, v in inputs.items() if hasattr(v, "to")}
            else:
                inputs = {"pixel_values": _to_tensor(arrs)}
            if hasattr(model, "get_image_features"):                 # CLIP / SigLIP / SigLIP2
                feats = model.get_image_features(**inputs)
            else:                                                     # DINOv2 / DINOv3 / ViT
                out = model(**inputs)
                feats = getattr(out, "pooler_output", None)
                if feats is None:
                    feats = out.last_hidden_state[:, 0]
        else:
            x = _to_tensor(arrs)
            if kind == "timm":
                feats = model(x)
            elif kind == "open_clip":
                feats = model.encode_image(x)
            else:
                feats = model(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
    return feats.cpu().numpy()


def _square(img, box, pad):
    """Crop (box ± pad) and letterbox to a square of STATE['size'] (keeps the whole object, no distortion)."""
    import numpy as np
    from PIL import Image
    W, H = img.size
    if box is not None:
        x1, y1, x2, y2 = box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        m = max(w, h) * pad
        x1, y1, x2, y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
        if x2 - x1 < 2 or y2 - y1 < 2:
            x1, y1, x2, y2 = max(0, x1 - 4), max(0, y1 - 4), min(W, x2 + 4), min(H, y2 + 4)
        img = img.crop((int(x1), int(y1), int(round(x2)), int(round(y2))))
    s = STATE["size"]
    w, h = img.size
    k = s / max(w, h, 1)
    img = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BICUBIC)
    canvas = Image.new("RGB", (s, s), (124, 116, 104))                 # ImageNet-mean grey
    canvas.paste(img, ((s - img.size[0]) // 2, (s - img.size[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def embed(items, out, batch=32, pad=0.1):
    import numpy as np
    from PIL import Image, ImageOps
    vecs, buf, failed = [], [], []

    def flush():
        if buf:
            vecs.append(_forward(buf).astype(np.float16))
            buf.clear()

    n = 0
    for it in items:
        boxes = it.get("boxes") or [None]
        try:
            img = ImageOps.exif_transpose(Image.open(it["image"])).convert("RGB")
        except Exception as exc:
            failed.append([n, len(boxes), str(exc)])
            img = None
        for b in boxes:
            buf.append(_square(img, b, pad) if img is not None else
                       np.zeros((STATE["size"], STATE["size"], 3), np.uint8))
            n += 1
            if len(buf) >= batch:
                flush()
    flush()
    arr = np.concatenate(vecs) if vecs else np.zeros((0, STATE["dim"] or 1), np.float16)
    np.save(out, arr)
    return {"n": int(arr.shape[0]), "failed": failed}


def knn(inp, k, out):
    import numpy as np
    x = np.load(inp)
    n = x.shape[0]
    k = max(1, min(int(k), n - 1)) if n > 1 else 0
    idx = np.zeros((n, max(k, 1)), np.int32)
    sim = np.zeros((n, max(k, 1)), np.float16)
    torch, done = None, False
    if k:
        try:
            import faiss
            xf = np.ascontiguousarray(x.astype(np.float32))
            index = faiss.IndexFlatIP(xf.shape[1])
            index.add(xf)
            s, i = index.search(xf, k + 1)
            idx[:], sim[:] = i[:, 1:], s[:, 1:]
            done = True
        except Exception:
            try:
                import torch
            except ImportError:
                torch = None
        if not done and torch is None:                                # numpy fallback (CPU, blocked)
            X = x.astype(np.float32)
            for a in range(0, n, 2048):
                s = X[a:a + 2048] @ X.T
                s[np.arange(s.shape[0]), np.arange(a, a + s.shape[0])] = -2
                part = np.argpartition(-s, k, axis=1)[:, :k]
                ps = np.take_along_axis(s, part, 1)
                order = np.argsort(-ps, axis=1)
                idx[a:a + 2048] = np.take_along_axis(part, order, 1)
                sim[a:a + 2048] = np.take_along_axis(ps, order, 1)
        elif not done:
            dev = STATE["device"] if STATE["model"] is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
            dt = torch.float16 if dev.startswith("cuda") else torch.float32
            X = torch.from_numpy(x).to(dev, dtype=dt)
            step = 4096 if dev.startswith("cuda") else 1024
            for a in range(0, n, step):
                s = X[a:a + step] @ X.T
                s[torch.arange(s.shape[0]), torch.arange(a, a + s.shape[0])] = -2      # not yourself
                v, i = torch.topk(s.float(), k, dim=1)
                idx[a:a + step] = i.cpu().numpy()
                sim[a:a + step] = v.cpu().numpy()
    np.save(out + "_idx.npy", idx)
    np.save(out + "_sim.npy", sim)
    return {"n": n, "k": k}


def download(repo, dest, token=None):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo, local_dir=dest, token=token or None,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.bin"])
    return {"path": path}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid, cmd = req.get("rid"), req.get("cmd")
        try:
            if cmd == "probe":
                reply(rid, ok=True, info=probe())
            elif cmd == "load":
                reply(rid, ok=True, **load(req["path"], req.get("device", "auto"), req.get("half", True)))
            elif cmd == "embed":
                reply(rid, ok=True, **embed(req["items"], req["out"], int(req.get("batch", 32)),
                                            float(req.get("pad", 0.1))))
            elif cmd == "knn":
                reply(rid, ok=True, **knn(req["inp"], int(req.get("k", 10)), req["out"]))
            elif cmd == "download":
                reply(rid, ok=True, **download(req["repo"], req["dest"], req.get("token")))
            elif cmd == "shutdown":
                reply(rid, ok=True)
                break
            else:
                reply(rid, ok=False, error=f"unknown command {cmd}")
        except Exception as exc:
            log(traceback.format_exc())
            reply(rid, ok=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
