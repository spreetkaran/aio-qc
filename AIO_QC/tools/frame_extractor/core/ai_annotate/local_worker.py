# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Runs INSIDE your Python environment (the one in Settings ▸ Models) — open-vocabulary detectors, local vision
models and SAM for AI annotate. Stand-alone on purpose: it imports nothing from AIO QC, so any environment with
PyTorch can run it.

Protocol: one JSON request per line on stdin, one JSON answer per line on stdout ({"id", "ok", ...} or
{"id", "ok": false, "error"}). Images travel as file paths (detect / segment) or base64 (vlm).

  probe                                 versions, CUDA, GPUs, which back-ends can load
  load_detector {type, model, device}   grounding_dino | owlv2 | florence2 | yolo_world | yolo | _test
  detect {path, crop, prompts, box_threshold, text_threshold}  -> boxes [[x1,y1,x2,y2,score,label]] (image px)
  load_vlm {model, device}              a Hugging Face vision-language model (AutoModelForImageTextToText),
                                        PaliGemma or Moondream
  vlm {system, parts, max_tokens}       -> text
  load_sam {model}                      SAM / SAM2 weights through ultralytics
  segment {path, boxes}                 -> polygons per box
"""
import base64
import importlib.util
import io
import json
import sys
import traceback

STATE = {}


def _has(m):
    try:
        return importlib.util.find_spec(m) is not None
    except (ImportError, ValueError):
        return False


def _ver(m):
    try:
        mod = __import__(m)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return None


def probe():
    info = {"python": sys.version.split()[0], "executable": sys.executable, "cuda": False, "devices": [],
            "packages": {m: _ver(m) if _has(m) else None for m in
                         ("torch", "torchvision", "transformers", "accelerate", "ultralytics", "timm", "einops",
                          "PIL", "cv2", "sam2", "qwen_vl_utils")}}
    if info["packages"]["torch"]:
        try:
            import torch
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    info["devices"].append(f"cuda:{i} {p.name} {p.total_memory / 2 ** 30:.1f} GB")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                info["devices"].append("mps")
        except Exception as exc:
            info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _device(d):
    import torch
    if d in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) is not None
                                                          and torch.backends.mps.is_available() else "cpu")
    return f"cuda:{d}" if str(d).isdigit() else d


def _image(path, crop=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if crop:
        x1, y1, x2, y2 = [int(round(v)) for v in crop]
        im = im.crop((x1, y1, x2, y2))
    return im


# ------------------------------------------------------------------------------------------------ detectors
def load_detector(type, model, device="auto", trust_remote_code=False, **_):
    if type == "_test":
        STATE["det"] = ("_test", None, None, "cpu")
        return {"labels": "any"}
    dev = _device(device)
    if type in ("grounding_dino", "owlv2"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        proc = AutoProcessor.from_pretrained(model)
        m = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(dev).eval()
        STATE["det"] = (type, m, proc, dev)
    elif type == "florence2":
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        dt = torch.float16 if dev.startswith("cuda") else torch.float32
        trc = bool(trust_remote_code)              # allowed by the user, per model (ai_annotate/trust.py)
        m = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=trc, torch_dtype=dt).to(dev).eval()
        proc = AutoProcessor.from_pretrained(model, trust_remote_code=trc)
        STATE["det"] = (type, m, proc, dev)
    elif type in ("yolo_world", "yolo"):
        from ultralytics import YOLO, YOLOWorld
        m = YOLOWorld(model) if type == "yolo_world" else YOLO(model)
        STATE["det"] = (type, m, None, dev)
    else:
        raise ValueError(f"unknown detector {type}")
    return {"device": dev}


def detect(path, prompts, crop=None, box_threshold=0.3, text_threshold=0.25, **_):
    kind, m, proc, dev = STATE["det"]
    im = _image(path, crop)
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    labels = list(prompts.keys())
    texts = [prompts[k] or k for k in labels]
    out = []
    if kind == "_test":       # for the app's own tests: one box per label, in the middle
        w, h = im.size
        for i, lab in enumerate(labels):
            out.append([w * 0.25 + i, h * 0.25, w * 0.75, h * 0.75, 0.9, lab])
    elif kind == "grounding_dino":
        import torch
        text = ". ".join(t.lower().rstrip(".") for t in texts) + "."
        inputs = proc(images=im, text=text, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = m(**inputs)
        kw = dict(target_sizes=[im.size[::-1]], text_threshold=text_threshold)
        try:
            r = proc.post_process_grounded_object_detection(o, inputs.input_ids, threshold=box_threshold, **kw)[0]
        except TypeError:
            r = proc.post_process_grounded_object_detection(o, inputs.input_ids, box_threshold=box_threshold, **kw)[0]
        names = r.get("text_labels", r.get("labels"))
        for b, s, n in zip(r["boxes"].tolist(), r["scores"].tolist(), names):
            n = str(n).lower()
            lab = next((labels[i] for i, t in enumerate(texts) if t.lower().rstrip(".") in n or n in t.lower()),
                       labels[0] if len(labels) == 1 else None)
            if lab:
                out.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, float(s), lab])
    elif kind == "owlv2":
        import torch
        inputs = proc(text=[texts], images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = m(**inputs)
        size = torch.tensor([im.size[::-1]])
        try:
            r = proc.post_process_grounded_object_detection(o, threshold=box_threshold, target_sizes=size)[0]
        except (AttributeError, TypeError):
            r = proc.post_process_object_detection(o, threshold=box_threshold, target_sizes=size)[0]
        for b, s, li in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()):
            out.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, float(s), labels[int(li)]])
    elif kind == "florence2":
        import torch
        for lab, t in zip(labels, texts):
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            inputs = proc(text=task + t, images=im, return_tensors="pt").to(dev, m.dtype)
            with torch.no_grad():
                ids = m.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 max_new_tokens=512, num_beams=3)
            txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
            r = proc.post_process_generation(txt, task=task, image_size=im.size)[task]
            for b in r.get("bboxes", []):
                out.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, 1.0, lab])
    elif kind in ("yolo_world", "yolo"):
        import numpy as np
        if kind == "yolo_world":
            m.set_classes(texts)
        res = m.predict(np.asarray(im)[:, :, ::-1], conf=box_threshold, device=dev, verbose=False)[0]
        names = res.names
        for b, s, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            n = labels[int(c)] if kind == "yolo_world" else str(names[int(c)])
            if kind == "yolo" and n not in labels and labels:
                continue
            out.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, float(s), n])
    return {"boxes": out}


# ------------------------------------------------------------------------------------------------ local VLM
def load_vlm(model, device="auto", trust_remote_code=False, **_):
    import torch
    trc = bool(trust_remote_code)                  # allowed by the user, per model (ai_annotate/trust.py)
    dev = _device(device)
    low = model.lower()
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    if "moondream" in low:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=trc, torch_dtype=dt).to(dev).eval()
        STATE["vlm"] = ("moondream", m, None, dev)
    else:
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:
            from transformers import AutoModelForVision2Seq as Auto
        proc = AutoProcessor.from_pretrained(model, trust_remote_code=trc)
        m = Auto.from_pretrained(model, torch_dtype=dt, trust_remote_code=trc).to(dev).eval()
        STATE["vlm"] = ("paligemma" if "paligemma" in low else "chat", m, proc, dev)
    return {"device": dev}


def vlm(system, parts, max_tokens=2048, **_):
    import torch
    from PIL import Image
    kind, m, proc, dev = STATE["vlm"]
    images, content = [], []
    for p in parts:
        if "text" in p:
            content.append({"type": "text", "text": p["text"]})
        elif "image" in p:
            im = Image.open(io.BytesIO(base64.b64decode(p["image"]))).convert("RGB")
            images.append(im)
            content.append({"type": "image"})
    if kind == "moondream":
        q = system + "\n" + " ".join(c.get("text", "") for c in content)
        return {"text": m.query(images[-1], q)["answer"]}
    msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
        [{"role": "user", "content": content}]
    try:
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    except Exception:
        prompt = system + "\n" + " ".join(c.get("text", "<image>") for c in content)
    inputs = proc(text=[prompt], images=images or None, return_tensors="pt").to(dev)
    with torch.no_grad():
        ids = m.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=False)
    new = ids[:, inputs["input_ids"].shape[1]:]
    return {"text": proc.batch_decode(new, skip_special_tokens=True)[0]}


# ------------------------------------------------------------------------------------------------ SAM
def load_sam(model, **_):
    from ultralytics import SAM
    STATE["sam"] = SAM(model)
    return {}


def segment(path, boxes, **_):
    import cv2
    res = STATE["sam"](path, bboxes=[b[:4] for b in boxes], verbose=False)[0]
    polys = []
    masks = res.masks.data.cpu().numpy() if res.masks is not None else []
    for mk in masks:
        cs, _ = cv2.findContours((mk > 0.5).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            polys.append(None)
            continue
        c = max(cs, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        polys.append(c.astype(float).tolist() if len(c) >= 3 else None)
    return {"polygons": polys + [None] * (len(boxes) - len(polys))}


OPS = {"probe": lambda **k: probe(), "load_detector": load_detector, "detect": detect, "load_vlm": load_vlm,
       "vlm": vlm, "load_sam": load_sam, "segment": segment}


def main():
    out = sys.stdout
    sys.stdout = sys.stderr          # libraries that print do not break the protocol
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.pop("id", None)
        op = req.pop("op", "")
        if op == "quit":
            break
        try:
            res = OPS[op](**req)
            msg = {"id": rid, "ok": True, **res}
        except Exception as exc:
            traceback.print_exc()
            msg = {"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(msg) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
