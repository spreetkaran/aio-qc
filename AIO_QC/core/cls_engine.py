# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC classifier trainer for any PyTorch architecture (timm or torchvision): ResNet, EfficientNet(V2), ConvNeXt(V2),
MobileNetV3/V4, RegNet, DenseNet, ViT, DeiT, Swin, EVA-02, DINOv2, FastViT, EfficientViT … — or any timm / torchvision
model by name. Runs INSIDE the user's Python environment and imports nothing from AIO QC: AIO QC copies it next to
train_runner.py (as aioqc_cls_engine.py), which hands over every spec with "engine": "torch".

    run(spec, status, ctrl, release)      train | resume | val — the same files as every other run:
                                          status.json, epochs.jsonl, results.csv, weights/{best,last}.pt

The model it saves (weights/best.pt) is a standard Ultralytics classification checkpoint: its "model" is an
nn.Sequential of the image normalisation (a fixed 1×1 convolution), the network and a softmax, carrying names / task /
imgsz — so YOLO("best.pt") predicts with it, and with it everything in AIO QC that uses a classifier (Testing, Model
Check, the Class Gallery's confidences, Frame Extractor pre-labels). weights/best.onnx is the same model for use
anywhere else (Ultralytics metadata inside: YOLO("best.onnx") works as well). weights/last.pt is the training state
(Resume continues exactly there).

Fine-tuning and improving: start from a model trained before (its best.pt) and the classes are matched by NAME — the
same classes keep what they learned, new ones get a fresh row, dropped ones go. Head-only epochs first (a linear probe,
the backbone frozen), layer-wise learning-rate decay, EMA of the weights, label smoothing, MixUp / CutMix, class-
balanced sampling or loss, hard-example sampling (images the starting model gets wrong are drawn more often).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

IMG_EXT = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".pfm", ".avif", ".heic",
           ".heif", ".jp2", ".gif"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VAL_DIRS = ("val", "valid", "validation", "test")
TRANSFORMER_WORDS = ("vit", "eva", "deit", "swin", "beit", "dinov2", "tiny_vit", "cait", "xcit", "pvt")
TV_FIXED = ("vit_", "swin", "maxvit")                  # torchvision transformers: their weights are for 224 only
FORMAT = "aioqc-cls 1"

# what a spec's args may hold (the app checks typed extra settings against this)
KEYS = {"data", "epochs", "imgsz", "batch", "device", "workers", "patience", "seed", "pretrained", "optimizer", "lr0",
        "lrf", "cos_lr", "warmup_epochs", "momentum", "weight_decay", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v",
        "scale", "degrees", "auto_augment", "erasing", "mixup", "cutmix", "label_smoothing", "dropout", "drop_path",
        "layer_decay", "ema", "freeze_epochs", "balance", "amp", "fraction", "time", "val_share", "onnx", "split",
        "project", "name", "exist_ok", "plots", "verbose", "cache", "close_mosaic", "grad_clip", "hard_boost",
        "interpolation", "probe_features", "grad_ckpt"}
DEFAULTS = {"epochs": 50, "imgsz": 224, "batch": 32, "device": "", "workers": 4, "patience": 15, "seed": 0,
            "pretrained": True, "optimizer": "auto", "lr0": 0.0, "lrf": 0.01, "cos_lr": True, "warmup_epochs": 3.0,
            "momentum": 0.9, "weight_decay": 0.05, "fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.015, "hsv_s": 0.4,
            "hsv_v": 0.4, "scale": 0.5, "degrees": 0.0, "auto_augment": "randaugment", "erasing": 0.25,
            "mixup": 0.0, "cutmix": 0.0, "label_smoothing": 0.1, "dropout": 0.0, "drop_path": 0.0,
            "layer_decay": 1.0, "ema": 0.9998, "freeze_epochs": 0, "balance": "none", "amp": True, "fraction": 1.0,
            "time": 0.0, "val_share": 0.15, "onnx": True, "grad_clip": 5.0, "hard_boost": 4.0,
            "probe_features": "cached", "grad_ckpt": "off"}


def _a(args, key):
    v = args.get(key)
    return DEFAULTS.get(key) if v is None else v


def probe() -> dict:
    out = {}
    for mod in ("timm", "torchvision"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


# ====================================================================================================== data
def _bucket(text: str, salt: str = "") -> float:
    """A stable number in [0, 1) for a file: the same image lands in the same split every run (and when images are
    added later, the old ones stay where they were)."""
    h = hashlib.md5((salt + text).encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def _walk(folder: Path) -> list:
    out = []
    for dp, dn, fn in os.walk(folder, followlinks=True):
        dn[:] = sorted(d for d in dn if not d.startswith("."))
        for f in sorted(fn):
            if os.path.splitext(f)[1].lower() in IMG_EXT and not f.startswith("."):
                out.append(os.path.join(dp, f))
    return out


def layout(root: Path) -> str:
    return "split" if (root / "train").is_dir() else "flat"


def gather(root, names, split: str, val_share: float = 0.15, fraction: float = 1.0) -> list:
    """[(path, class index)] of one split. A folder with train/ (and val/ or valid/ / test/) is read as it is; a flat
    folder of class folders is split here — `val_share` of each class to val, by a stable hash of the file name."""
    root = Path(root)
    items = []
    if layout(root) == "split":
        if split == "val":
            base = next((root / d for d in VAL_DIRS if (root / d).is_dir()), None)
        else:
            base = root / split if (root / split).is_dir() else None
        if base is None:
            return []
        for i, n in enumerate(names):
            if (base / n).is_dir():
                items += [(p, i) for p in _walk(base / n)]
    else:
        if split == "test":
            return []
        for i, n in enumerate(names):
            files = _walk(root / n) if (root / n).is_dir() else []
            keys = [(_bucket(os.path.relpath(p, root).replace("\\", "/").lower()), p) for p in files]
            val = {p for b, p in keys if b < val_share}
            if not val and len(files) >= 2 and val_share > 0:     # every class with 2+ images gets a val image:
                val = {min(keys)[1]}                              # its lowest (a tiny class; may change as it grows)
            items += [(p, i) for p in files if (p in val) == (split == "val")]
    if split == "train" and fraction < 1.0:
        items = [it for it in items if _bucket(it[0], "fraction") < fraction] or items[:1]
    return items


class FolderData:
    """(image tensor, class index). A file that can't be read is replaced by another image (and reported once)."""

    def __init__(self, items, transform, recipe=None):
        self.items = items
        self.transform = transform
        self.recipe = recipe
        self._warned = False

    def __len__(self):
        return len(self.items)

    def _open(self, path):
        from PIL import Image
        with Image.open(path) as im:
            return im.convert("RGB")

    def __getitem__(self, i):
        for attempt in range(6):
            path, label = self.items[i]
            try:
                im = self._open(path)
                if self.recipe is not None:
                    im = self.recipe(im)
                return self.transform(im), label
            except Exception as exc:
                if not self._warned:
                    self._warned = True
                    print(f"WARNING: could not read {path} ({type(exc).__name__}: {exc}) — another image is used",
                          flush=True)
                i = random.randrange(len(self.items))
        raise RuntimeError("several images in a row could not be read")


def transforms(args, mean, std, train: bool, size: int):
    """Training: random-resized crop, flips, rotation, an auto-augment policy (or colour jitter), random erasing.
    Evaluation — the same as Ultralytics' classify_transforms, so scores here match predictions anywhere: the shorter
    side resized to `size`, the centre cropped, no normalisation (the saved model does it itself)."""
    import torchvision.transforms as T
    bil = T.InterpolationMode.BILINEAR
    if not train:
        return T.Compose([T.Resize(size, interpolation=bil), T.CenterCrop(size), T.ToTensor(),
                          T.Normalize(mean, std)])
    tfl = []
    deg = float(_a(args, "degrees") or 0)
    if deg > 0:
        tfl.append(T.RandomRotation(deg, interpolation=bil))
    sc = float(_a(args, "scale") or 0)
    if sc > 0:
        tfl.append(T.RandomResizedCrop(size, scale=(max(0.02, 1.0 - sc), 1.0), interpolation=bil, antialias=True))
    else:
        tfl += [T.Resize(size, interpolation=bil), T.CenterCrop(size)]
    if float(_a(args, "fliplr")) > 0:
        tfl.append(T.RandomHorizontalFlip(float(_a(args, "fliplr"))))
    if float(_a(args, "flipud")) > 0:
        tfl.append(T.RandomVerticalFlip(float(_a(args, "flipud"))))
    pol = str(_a(args, "auto_augment") or "").lower().replace("_", "").replace("-", "")
    if pol in ("randaugment", "randaug"):
        tfl.append(T.RandAugment(num_ops=2, magnitude=9, interpolation=bil))
    elif pol == "autoaugment":
        tfl.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, interpolation=bil))
    elif pol == "augmix":
        tfl.append(T.AugMix(interpolation=bil))
    elif pol in ("trivialaugment", "trivialaugmentwide", "trivial"):
        tfl.append(T.TrivialAugmentWide(interpolation=bil))
    else:                                              # no policy: colour jitter from the hue / saturation / value
        h, s, v = (float(_a(args, k)) for k in ("hsv_h", "hsv_s", "hsv_v"))
        if h or s or v:
            tfl.append(T.ColorJitter(brightness=v, contrast=v, saturation=s, hue=min(0.5, h)))
    tfl += [T.ToTensor(), T.Normalize(mean, std)]
    er = float(_a(args, "erasing") or 0)
    if er > 0:
        tfl.append(T.RandomErasing(p=er, inplace=True))
    return T.Compose(tfl)


# ====================================================================================================== model
def _tv_head(net):
    """(dotted name, module) of a torchvision model's last Linear layer — its classifier."""
    import torch.nn as nn
    last = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            last = (name, m)
    if last is None:
        raise RuntimeError("no Linear classifier layer found in this model")
    return last


def _set_module(net, dotted, new):
    parent = net
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def head(net, info):
    """The classifier layer (Linear) — what changes with the classes."""
    if info.get("lib") == "timm" and hasattr(net, "get_classifier"):
        m = net.get_classifier()
        if m is not None and hasattr(m, "weight"):
            return m
    return _tv_head(net)[1]


def set_classes(net, info, nc: int):
    """A new, untrained classifier for nc classes."""
    import torch.nn as nn
    if info.get("lib") == "timm" and hasattr(net, "reset_classifier"):
        net.reset_classifier(nc)
        return
    name, old = _tv_head(net)
    _set_module(net, name, nn.Linear(old.in_features, nc, bias=old.bias is not None))


def fixed_size(net) -> int | None:
    """A transformer whose position embedding fixes its input size (None: any size works)."""
    pe = getattr(net, "patch_embed", None)
    fixed = getattr(pe, "img_size", None)
    if not fixed or getattr(pe, "dynamic_img_size", False):
        return None
    fixed = fixed[0] if isinstance(fixed, (tuple, list)) else fixed
    return int(fixed)


def snap(size: int, patch: int = 0) -> int:
    """Image sizes are multiples of 32 — Ultralytics rounds any other up when it predicts, and the exported ONNX
    has a fixed size — and of a transformer's patch size (14: multiples of 224)."""
    step = 32 if not patch or 32 % patch == 0 else 32 * patch // math.gcd(32, patch)
    return max(step, int(round(int(size) / float(step))) * step)


def patch_of(name: str) -> int:
    import re
    m = re.search(r"patch(\d+)", str(name))
    return int(m.group(1)) if m else 0


def create(model: str, nc: int, args, status=None):
    """"timm:<name>" or "tv:<name>" (a torchvision model) -> (net, info). Pretrained weights are downloaded once
    (timm: Hugging Face; torchvision: download.pytorch.org)."""
    import torch.nn as nn
    lib, _, name = model.partition(":")
    pretrained = bool(_a(args, "pretrained"))
    size = snap(int(_a(args, "imgsz")), patch_of(model))
    if size != snap(int(_a(args, "imgsz"))):
        print(f"{name}: {size} px (a multiple of its {patch_of(model)}-pixel patches and of 32)", flush=True)
    drop, dpr = float(_a(args, "dropout") or 0), float(_a(args, "drop_path") or 0)
    info = {"lib": "timm" if lib == "timm" else "torchvision", "arch": name, "imgsz": size}
    if lib == "timm":
        import timm
        kw = {"pretrained": pretrained, "num_classes": nc}
        if drop:
            kw["drop_rate"] = drop
        if dpr:
            kw["drop_path_rate"] = dpr
        wants_size = any(w in name.lower() for w in TRANSFORMER_WORDS)
        tries = ([dict(kw, img_size=size)] if wants_size else []) + [kw, {"pretrained": pretrained, "num_classes": nc}]
        net, err = None, None
        for k in tries:
            try:
                net = timm.create_model(name, **k)
                break
            except (TypeError, AssertionError, ValueError, RuntimeError) as exc:
                err = exc                                 # no img_size / drop_path here, or not at this size
                if "img_size" in k:
                    print(f"{name} can't run at {size} px ({type(exc).__name__}: {exc}) — its own size is used",
                          flush=True)
        if net is None:
            raise err
        cfg = getattr(net, "pretrained_cfg", None) or {}
        info["mean"] = list(cfg.get("mean") or IMAGENET_MEAN)
        info["std"] = list(cfg.get("std") or IMAGENET_STD)
        fixed = fixed_size(net)
        if fixed and fixed != size:
            print(f"{name} runs at {fixed} px (its fixed input size), not {size}", flush=True)
            info["imgsz"] = fixed
        return net, info
    import torchvision
    if any(name.startswith(p) for p in TV_FIXED) and size != 224:
        print(f"{name} (torchvision) runs at 224 px — its weights are for that size only", flush=True)
        info["imgsz"] = 224
    get = getattr(torchvision.models, "get_model", None)
    if get is not None:
        net = get(name, weights="DEFAULT" if pretrained else None)
    else:                                              # an older torchvision
        net = getattr(torchvision.models, name)(pretrained=pretrained)
    set_classes(net, info, nc)
    if drop:
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop
    info["mean"], info["std"] = list(IMAGENET_MEAN), list(IMAGENET_STD)
    return net, info


def load(path):
    """An AIO QC classifier file -> (net, info, names, ckpt). best.pt (the Sequential: normalisation, network,
    softmax) or last.pt (the training state)."""
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} is not a model file")
    if ckpt.get("format") == FORMAT and "net" in ckpt:           # last.pt: the training state
        info = dict(ckpt["info"])
        return ckpt["net"].float(), info, list(info["names"]), ckpt
    info = ckpt.get("aioqc")
    if not info:
        raise ValueError(f"{Path(path).name} was not trained by AIO QC's PyTorch trainer — YOLO classifiers train "
                         "with Ultralytics (choose a YOLO …-cls model, or this file there)")
    seq = ckpt.get("ema") or ckpt["model"]
    return seq[1].float(), dict(info), list(info["names"]), ckpt


def remap(net, info, old_names, new_names) -> dict:
    """The classifier rows follow the class NAMES: a class the model knew keeps its row, a new class gets a fresh
    one. -> {"kept": [...], "new": [...], "dropped": [...]}."""
    import torch
    old = head(net, info)
    W = old.weight.detach().clone()
    b = old.bias.detach().clone() if old.bias is not None else None
    if list(old_names) == list(new_names):
        return {"kept": list(new_names), "new": [], "dropped": []}
    set_classes(net, info, len(new_names))
    new = head(net, info)
    kept = []
    with torch.no_grad():
        for i, n in enumerate(new_names):
            if n in old_names:
                j = old_names.index(n)
                new.weight[i].copy_(W[j].to(new.weight.dtype).view_as(new.weight[i]))
                if b is not None and new.bias is not None:
                    new.bias[i] = b[j]
                kept.append(n)
    return {"kept": kept, "new": [n for n in new_names if n not in old_names],
            "dropped": [n for n in old_names if n not in new_names]}


def deploy(net, info, names, size):
    """The network as an Ultralytics classification model: normalisation (a fixed 1×1 conv) → network → softmax,
    with names / task / stride / transforms attached. Standard torch modules only (and the library's own network
    class), so any Python with torch (+ timm for timm models) and Ultralytics can load it."""
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    mean, std = info.get("mean") or IMAGENET_MEAN, info.get("std") or IMAGENET_STD
    norm = nn.Conv2d(3, 3, 1, groups=3, bias=True)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0 / s for s in std]).view(3, 1, 1, 1))
        norm.bias.copy_(torch.tensor([-m / s for m, s in zip(mean, std)]))
    norm.requires_grad_(False)
    body = copy.deepcopy(net).float().cpu()
    if hasattr(body, "set_grad_checkpointing"):           # (a training-time memory saver: off in the saved model)
        try:
            body.set_grad_checkpointing(False)
        except Exception:
            pass
    seq = nn.Sequential(norm, body, nn.Softmax(dim=1)).eval()
    seq.names = {i: str(n) for i, n in enumerate(names)}
    seq.task = "classify"
    seq.stride = torch.tensor([32.0])
    seq.yaml = {"channels": 3, "nc": len(names), "aioqc": info.get("arch")}
    seq.args = {"task": "classify", "imgsz": int(size)}
    seq.transforms = T.Compose([T.Resize(int(size), interpolation=T.InterpolationMode.BILINEAR),
                                T.CenterCrop(int(size)), T.ToTensor(),
                                T.Normalize(torch.tensor((0.0, 0.0, 0.0)), torch.tensor((1.0, 1.0, 1.0)))])
    return seq


def _atomic_save(obj, path: Path):
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, str(tmp))
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.05)


def save_best(path: Path, net, info, names, size, extra: dict):
    seq = deploy(net, info, names, size).half()           # one copy of the network (large models: memory)
    ck = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": FORMAT, "model": seq,
          "ema": None, "optimizer": None, "train_args": {"task": "classify", "imgsz": int(size),
                                                         "model": f"{info['lib']}:{info['arch']}",
                                                         "data": extra.get("data")},
          "epoch": extra.get("epoch", -1), "best_fitness": extra.get("top1"),
          "aioqc": dict(info, names=list(names), imgsz=int(size), **{k: extra[k] for k in ("top1", "top5", "epoch")
                                                                      if k in extra})}
    _atomic_save(ck, path)
    side = {"engine": "torch", "lib": info["lib"], "arch": info["arch"], "names": list(names), "imgsz": int(size),
            **{k: extra[k] for k in ("top1", "top5", "epoch") if k in extra}}
    path.with_name(path.stem + ".aioqc.json").write_text(json.dumps(side, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    return seq


# ------------------------------------------------------------------ ONNX
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(no: int, data: bytes) -> bytes:
    return _varint(no << 3 | 2) + _varint(len(data)) + data


def add_onnx_metadata(path, meta: dict):
    """Key / value strings into an .onnx file's metadata_props (ModelProto field 14) — appended to the file, which
    protobuf reads as more entries of that repeated field. Needs no onnx package."""
    raw = b"".join(_field(14, _field(1, str(k).encode("utf-8")) + _field(2, str(v).encode("utf-8")))
                   for k, v in meta.items())
    with open(path, "ab") as f:
        f.write(raw)


def export_onnx(seq, path: Path, size: int, names, info) -> Path:
    """The deployed model as ONNX (dynamic batch, opset 17) with the metadata Ultralytics reads (task, names,
    imgsz): YOLO("best.onnx") and onnxruntime both run it."""
    import torch
    m = copy.deepcopy(seq).float().cpu().eval()
    x = torch.zeros(1, 3, int(size), int(size))
    kw = dict(input_names=["images"], output_names=["output0"],
              dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}}, opset_version=17)
    tmp = path.with_name(path.stem + ".tmp.onnx")
    try:
        torch.onnx.export(m, x, str(tmp), dynamo=False, **kw)
    except TypeError:                                  # an older torch: no dynamo switch
        torch.onnx.export(m, x, str(tmp), **kw)
    add_onnx_metadata(tmp, {"description": f"AIO QC classifier ({info['lib']} {info['arch']})", "author": "AIO QC",
                            "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": FORMAT, "task": "classify",
                            "stride": 32, "batch": 1, "channels": 3, "imgsz": str([int(size), int(size)]),
                            "names": str({i: str(n) for i, n in enumerate(names)}),
                            "args": str({"batch": 1, "half": False, "dynamic": True, "nms": False})})
    os.replace(tmp, path)
    return path


# ====================================================================================================== training
class EMA:
    """An exponential moving average of the weights (it ramps up over the first updates, as Ultralytics' does)."""

    def __init__(self, net, decay: float, tau: float = 2000.0, updates: int = 0):
        import torch
        self.module = copy.deepcopy(net).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = float(decay), float(tau), int(updates)
        self._torch = torch

    def update(self, net):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        with self._torch.no_grad():
            msd = net.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
                else:
                    v.copy_(msd[k])


def _device(args):
    import torch
    d = str(_a(args, "device") or "").strip().lower()
    if d in ("", "auto", "none"):
        d = "0" if torch.cuda.is_available() else "cpu"
    if d == "mps":
        return torch.device("mps")
    if d == "cpu" or not torch.cuda.is_available():
        if d not in ("cpu", "mps"):
            print(f"No CUDA GPU here — training on the CPU (asked for device {d})", flush=True)
        return torch.device("cpu")
    first = d.split(",")[0].replace("cuda:", "")
    if "," in d:
        print(f"One GPU is used ({first}); several GPUs train YOLO classifiers (Ultralytics) only", flush=True)
    return torch.device(f"cuda:{int(first)}")


def _optimizer(net, info, args, lr, batch):
    import torch
    name = str(_a(args, "optimizer") or "auto").lower()
    if name in ("auto", "musgd"):
        name = "adamw"
    wd = float(_a(args, "weight_decay"))
    mom = float(_a(args, "momentum"))
    ld = float(_a(args, "layer_decay") or 1.0)
    if info.get("lib") == "timm" and ld < 1.0 and hasattr(net, "group_matcher"):
        try:
            from timm.optim import create_optimizer_v2
            return create_optimizer_v2(net, opt=name, lr=lr, weight_decay=wd, momentum=mom, layer_decay=ld), \
                f"{name} with layer-wise decay {ld}"
        except Exception as exc:
            print(f"Layer-wise decay not used ({type(exc).__name__}: {exc})", flush=True)
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=mom, nesterov=True), "SGD"
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=(mom, 0.999)), "Adam"
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, alpha=0.9, momentum=mom, eps=0.0316), "RMSprop"
    return torch.optim.AdamW(groups, lr=lr, betas=(mom, 0.999)), "AdamW"


def auto_lr(args, batch, pretrained_start: bool) -> float:
    lr = float(_a(args, "lr0") or 0)
    if lr > 0:
        return lr
    name = str(_a(args, "optimizer") or "auto").lower()
    scale = max(batch, 1) / 64.0
    if name in ("sgd", "musgd"):
        return round(0.01 * scale, 6)
    if name == "rmsprop":
        return round(0.001 * scale, 6)
    return round((0.0005 if pretrained_start else 0.002) * math.sqrt(scale), 6)


def lr_factor(step, total, warm, lrf, cos):
    if warm and step < warm:
        return 0.01 + 0.99 * step / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    if cos:
        return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * p))
    return 1 - (1 - lrf) * p


def _soft_ce(logits, target, weight=None):
    import torch
    loss = -(target * torch.log_softmax(logits.float(), dim=1)).sum(dim=1)
    if weight is not None:
        loss = loss * (target * weight).sum(dim=1)
    return loss.mean()


def _mix(x, y_soft, mixup: float, cutmix: float, rng):
    """MixUp (blend two images) or CutMix (paste a patch of another): each batch has the chance given."""
    import torch
    r = rng.random()
    if r >= mixup + cutmix or x.shape[0] < 2:
        return x, y_soft
    perm = torch.randperm(x.shape[0], device=x.device)
    if r < mixup:
        lam = float(rng.beta(0.8, 0.8))
        x = x * lam + x[perm] * (1 - lam)
    else:
        lam = float(rng.beta(1.0, 1.0))
        H, W = x.shape[2], x.shape[3]
        ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / float(H * W)
    return x, y_soft * lam + y_soft[perm] * (1 - lam)


def evaluate(net, loader, device, nc, amp=False, keep=False):
    """-> {"top1", "top5", "loss", "cm" (rows: predicted, columns: true), "n", "rows" [(index, pred, conf, p_true)]}"""
    import torch
    net.eval()
    k = min(5, nc)
    cm = [[0] * nc for _ in range(nc)]
    n = hit1 = hit5 = 0
    loss_sum = 0.0
    rows = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = net(x)
            out = out.float()
            loss_sum += float(torch.nn.functional.cross_entropy(out, y, reduction="sum"))
            prob = out.softmax(1)
            top = prob.topk(k, dim=1).indices
            hit1 += int((top[:, 0] == y).sum())
            hit5 += int((top == y[:, None]).any(dim=1).sum())
            for p_, t_ in zip(top[:, 0].tolist(), y.tolist()):
                cm[p_][t_] += 1
            if keep:
                conf = prob.max(dim=1)
                pt = prob.gather(1, y[:, None])[:, 0]
                for j in range(len(y)):
                    rows.append((n + j, int(conf.indices[j]), float(conf.values[j]), float(pt[j])))
            n += len(y)
    return {"top1": hit1 / max(n, 1), "top5": hit5 / max(n, 1), "loss": loss_sum / max(n, 1), "cm": cm, "n": n,
            "rows": rows}


def per_class(cm, names, counts=None):
    """Rows the run view shows (under the box_* keys): precision, recall (= the class's accuracy)."""
    out = []
    nc = len(names)
    for c in range(nc):
        tp = cm[c][c]
        col = sum(cm[r][c] for r in range(nc))
        row = sum(cm[c])
        r = tp / col if col else 0.0
        p = tp / row if row else 0.0
        out.append({"id": c, "name": str(names[c]), "box_p": round(p, 4), "box_r": round(r, 4),
                    "box_map50": round(r, 4), "box_map": round(r, 4), "instances": int(col),
                    **({"train_images": int(counts.get(c, 0))} if counts is not None else {})})
    return out


def _loader(ds, batch, workers, shuffle=False, sampler=None, drop_last=False, device=None):
    import torch
    w = max(0, min(int(workers), os.cpu_count() or 1))
    return torch.utils.data.DataLoader(ds, batch_size=int(batch), shuffle=shuffle and sampler is None,
                                       sampler=sampler, num_workers=w, pin_memory=device is not None and
                                       device.type == "cuda", drop_last=drop_last, persistent_workers=w > 0)


def _auto_batch(net, device, size, n_train, amp, status):
    """The largest batch (8 … 256) whose training step fits in 85 % of the GPU's memory."""
    import torch
    if device.type != "cuda":
        return 32 if device.type == "mps" else 16
    status.set(phase="finding the batch size that fits the GPU")
    total = torch.cuda.get_device_properties(device).total_memory
    good, b = 8, 8
    net.train()
    while b <= 256 and b <= max(8, n_train // 2):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(b, 3, size, size, device=device)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = net(x)
            out.float().mean().backward()
            net.zero_grad(set_to_none=True)
            if torch.cuda.max_memory_reserved(device) > 0.85 * total:
                break
            good = b
            b *= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            break
        finally:
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    print(f"Batch size: {good} (the largest that fits the GPU)", flush=True)
    return good


def _set_trainable(net, info, head_only: bool):
    h = head(net, info)
    hp = {id(p) for p in h.parameters()}
    for p in net.parameters():
        p.requires_grad_(not head_only or id(p) in hp)


def _write_results_csv(run: Path, rows):
    keys = ["epoch", "time", "train/loss", "metrics/accuracy_top1", "metrics/accuracy_top5", "val/loss", "lr/pg0"]
    tmp = run / "results.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r["epoch"], r.get("time", 0), r["loss"].get("train/loss"),
                        r["metrics"].get("metrics/accuracy_top1"), r["metrics"].get("metrics/accuracy_top5"),
                        r["metrics"].get("val/loss"), r["lr"].get("lr/pg0")])
    os.replace(tmp, run / "results.csv")


def _write_reports(run: Path, names, items, ev):
    """confusion_matrix.csv (rows: true class, columns: predicted) and val_predictions.csv (every val image: its
    class, the prediction, the confidence) — the mistakes are what to look at next."""
    nc = len(names)
    with open(run / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + list(names))
        for t in range(nc):
            w.writerow([names[t]] + [ev["cm"][p][t] for p in range(nc)])
    with open(run / "val_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "predicted", "confidence", "correct", "p_class"])
        for i, pred, conf, pt in sorted(ev["rows"], key=lambda r: (r[3], r[0])):
            path, t = items[i]
            w.writerow([path, names[t], names[pred], round(conf, 4), int(pred == t), round(pt, 4)])


def _features(net, info, items, tf, device, amp, batch, workers, status, what):
    """What the classifier layer sees for every image (its input, caught by a hook): (N, D) on the CPU."""
    import torch
    h = head(net, info)
    got = []
    hook = h.register_forward_pre_hook(lambda _m, inp: got.append(inp[0].detach().float().flatten(1).cpu()))
    dl = _loader(FolderData(items, tf), batch, workers, device=device)
    n, last = len(dl), 0.0
    net.eval()
    try:
        with torch.no_grad():
            for i, (x, _y) in enumerate(dl):
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    net(x)
                if time.time() - last > 1.0 or i + 1 == n:
                    last = time.time()
                    status.set(phase=f"{what} ({i + 1}/{n} batches)", batch=i + 1, batches=n)
    finally:
        hook.remove()
    return torch.cat(got) if got else torch.zeros(0, 1)


def _scores(logits, Y, nc, keep=False):
    """evaluate()'s result from logits already computed."""
    import torch
    out = logits.float()
    k = min(5, nc)
    prob = out.softmax(1)
    top = prob.topk(k, dim=1).indices
    n = len(Y)
    cm = [[0] * nc for _ in range(nc)]
    for p_, t_ in zip(top[:, 0].tolist(), Y.tolist()):
        cm[p_][t_] += 1
    rows = []
    if keep:
        conf = prob.max(dim=1)
        pt = prob.gather(1, Y[:, None])[:, 0]
        rows = [(j, int(conf.indices[j]), float(conf.values[j]), float(pt[j])) for j in range(n)]
    return {"top1": float((top[:, 0] == Y).float().mean()) if n else 0.0,
            "top5": float((top == Y[:, None]).any(dim=1).float().mean()) if n else 0.0,
            "loss": float(torch.nn.functional.cross_entropy(out, Y)) if n else 0.0, "cm": cm, "n": n, "rows": rows}


def linear_probe(net, info, names, train_items, val_items, eval_tf, args, device, amp, batch, workers, status, ctrl,
                 item_w, class_w, wdir, run, root, t0):
    """Few labelled images: the backbone stays as it is, every image goes through it once (a mirrored copy too when
    mirroring is on), and only the classifier layer learns, on those features — seconds per epoch for any model
    size. -> (best top-1, best epoch, epoch rows, why it stopped)."""
    import torch
    import torchvision.transforms as TT
    nc = len(names)
    epochs = int(_a(args, "epochs"))
    views = [("", eval_tf)]
    if float(_a(args, "fliplr") or 0) > 0:
        views.append((" (mirrored)", TT.Compose([eval_tf, TT.RandomHorizontalFlip(1.0)])))
    if float(_a(args, "flipud") or 0) > 0:
        views.append((" (upside-down)", TT.Compose([eval_tf, TT.RandomVerticalFlip(1.0)])))
    fb = max(int(batch), 32)
    t_feat = time.time()
    Xtr = torch.cat([_features(net, info, train_items, tf, device, amp, fb, workers, status,
                               f"features of the training images{tag}") for tag, tf in views])
    Ytr = torch.tensor([c for _p, c in train_items]).repeat(len(views))
    Wtr = torch.tensor(item_w, dtype=torch.float64).repeat(len(views)) if item_w is not None else None
    Xva = _features(net, info, val_items, eval_tf, device, amp, fb, workers, status, "features of the val images")
    Yva = torch.tensor([c for _p, c in val_items])
    print(f"Linear probe: {len(Xtr):,} training features ({len(views)} view(s)) and {len(Xva):,} val, "
          f"{Xtr.shape[1]} values each, in {time.time() - t_feat:.0f} s — now only the classifier layer learns",
          flush=True)
    hd = device if device.type == "cuda" and (Xtr.numel() + Xva.numel()) * 4 < 2e9 else torch.device("cpu")
    Xtr, Ytr, Xva, Yva = Xtr.to(hd), Ytr.to(hd), Xva.to(hd), Yva.to(hd)
    h = head(net, info)
    hc = copy.deepcopy(h).float().to(hd).train()
    cw = class_w.to(hd) if class_w is not None else None
    lr = float(_a(args, "lr0") or 0)
    name = str(_a(args, "optimizer") or "auto").lower()
    wd = float(_a(args, "weight_decay"))
    if name in ("sgd", "musgd"):
        opt = torch.optim.SGD(hc.parameters(), lr=lr or 0.05, momentum=float(_a(args, "momentum")), weight_decay=wd)
    else:
        opt = torch.optim.AdamW(hc.parameters(), lr=lr or 0.001, weight_decay=wd)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    bs = max(1, min(len(Xtr), max(256, int(batch))))
    iters = math.ceil(len(Xtr) / bs)
    total, warm = epochs * iters, int(float(_a(args, "warmup_epochs") or 0) * iters)
    lrf, cos = float(_a(args, "lrf")), bool(_a(args, "cos_lr"))
    ls = float(_a(args, "label_smoothing") or 0)
    patience = int(_a(args, "patience") or 0)
    limit = float(_a(args, "time") or 0) * 3600
    best, best_epoch, best_sd, rows, step, stopped_by = -1.0, 0, None, [], 0, ""
    status.set(phase="training the classifier layer on the features", batches=iters)
    for epoch in range(epochs):
        t_ep = time.time()
        order = (torch.multinomial(Wtr, len(Xtr), replacement=True) if Wtr is not None
                 else torch.randperm(len(Xtr))).to(hd)
        loss_sum = 0.0
        for b in range(iters):
            f = lr_factor(step, total, warm, lrf, cos)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            j = order[b * bs:(b + 1) * bs]
            soft = torch.nn.functional.one_hot(Ytr[j], nc).float()
            if ls:
                soft = soft * (1 - ls) + ls / nc
            loss = _soft_ce(hc(Xtr[j]), soft, cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += float(loss.detach()) * len(j)
        hc.eval()
        with torch.no_grad():
            ev = _scores(torch.cat([hc(Xva[i:i + 4096]) for i in range(0, len(Xva), 4096)]), Yva, nc)
        hc.train()
        if ev["top1"] > best:
            best, best_epoch = ev["top1"], epoch + 1
            best_sd = copy.deepcopy(hc.state_dict())
        row = {"epoch": epoch + 1, "epochs": epochs, "loss": {"train/loss": round(loss_sum / len(Xtr), 5)},
               "metrics": {"metrics/accuracy_top1": round(ev["top1"], 5), "metrics/accuracy_top5": round(ev["top5"], 5),
                           "val/loss": round(ev["loss"], 5)},
               "lr": {"lr/pg0": round(opt.param_groups[0]["lr"], 8)}, "time": round(time.time() - t_ep, 3),
               "fitness": round(ev["top1"], 5), "best_fitness": round(best, 5), "final": False, "t": time.time()}
        rows.append(row)
        status.event(row)
        status.set(metrics=row["metrics"], best_fitness=round(best, 5), last_epoch=epoch + 1, epoch=epoch + 1,
                   elapsed=round(time.time() - t0, 1), best_epoch=best_epoch, batch=iters)
        if ctrl.get("stop") or ctrl.get("pause"):
            stopped_by = "stop"
            break
        if patience and epoch + 1 - best_epoch >= patience:
            stopped_by = "patience"
            print(f"Stopping: no better val accuracy for {patience} epochs (best: epoch {best_epoch}, top-1 "
                  f"{best:.4f})", flush=True)
            break
        if limit and time.time() - t0 > limit:
            stopped_by = "time"
            break
    _write_results_csv(run, rows)
    if best_sd is not None:
        h.load_state_dict(best_sd)
    status.set(phase="saving best.pt")
    save_best(wdir / "best.pt", net, info, names, info["imgsz"],
              {"top1": round(best, 5), "epoch": best_epoch, "data": str(root)})
    _atomic_save({"format": FORMAT, "net": copy.deepcopy(net).float().cpu(), "info": info, "args": dict(args),
                  "optimizer": None, "scaler": None, "ema": None, "ema_updates": 0, "epoch": len(rows) - 1,
                  "best": best, "best_epoch": best_epoch, "step": step, "rows": rows, "probe": True}, wdir / "last.pt")
    status.set(best=str(wdir / "best.pt"), last=str(wdir / "last.pt"))
    return best, best_epoch, rows, stopped_by


def train(spec, status, ctrl, release):
    import numpy as np
    import torch
    mode = spec.get("mode", "train")
    state = None
    if mode == "resume":
        state = torch.load(spec["resume_from"], map_location="cpu", weights_only=False)
        args = dict(state["args"])
    else:
        args = dict(spec["args"])
    meta = Path(spec["_meta"])
    run = meta.parent
    wdir = run / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    names = list(state["info"]["names"] if state else spec["names"])
    nc = len(names)
    seed = int(_a(args, "seed"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + (state["epoch"] + 1 if state else 0))
    device = _device(args)
    amp = bool(_a(args, "amp")) and device.type == "cuda"

    # ---------------------------------------------------------------- the model
    status.set(state="starting", phase="loading the model")
    remapped = None
    start_trained = False
    if state is not None:
        net, info = state["net"].float(), dict(state["info"])
    else:
        src = str(spec["model"])
        if src.lower().endswith((".pt", ".pth")):
            net, info, old_names, _ck = load(src)
            info = {k: info[k] for k in ("lib", "arch", "mean", "std", "imgsz") if k in info}
            remapped = remap(net, info, old_names, names)
            start_trained = True
            if remapped["new"] or remapped["dropped"]:
                print(f"Classes matched by name: {len(remapped['kept'])} kept"
                      + (f", new: {', '.join(remapped['new'][:10])}" if remapped["new"] else "")
                      + (f", dropped: {', '.join(remapped['dropped'][:10])}" if remapped["dropped"] else ""),
                      flush=True)
            want = snap(int(_a(args, "imgsz")), patch_of(info.get("arch")))
            fixed = fixed_size(net) if info.get("lib") == "timm" else (
                int(info.get("imgsz") or 224) if any(info.get("arch", "").startswith(p) for p in TV_FIXED) else None)
            if fixed and fixed != want:
                print(f"{info.get('arch')} runs at {fixed} px (its fixed input size), not {want}", flush=True)
            info["imgsz"] = fixed or want
        else:
            net, info = create(src, nc, args, status)
    size = snap(int(info.get("imgsz") or _a(args, "imgsz")), patch_of(info.get("arch")))
    info["imgsz"] = size
    info["names"] = names
    mean, std = info.get("mean") or IMAGENET_MEAN, info.get("std") or IMAGENET_STD
    net = net.to(device)
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)

    # ---------------------------------------------------------------- the data
    status.set(phase="reading the image list")
    root = Path(args["data"])
    vs, frac = float(_a(args, "val_share")), float(_a(args, "fraction"))
    train_items = gather(root, names, "train", vs, frac)
    val_items = gather(root, names, "val", vs)
    if not train_items:
        raise RuntimeError(f"No training images found in {root} for the classes {', '.join(names[:6])} …")
    if not val_items:
        raise RuntimeError(f"No validation images found in {root} (val/ or valid/ or test/, or a flat folder of "
                           "class folders with 2+ images per class)")
    if layout(root) == "flat":
        (meta / "val_images.txt").write_text("\n".join(p for p, _c in val_items) + "\n", encoding="utf-8")
        print(f"No train/val folders: {len(val_items):,} images ({vs:.0%} of each class, the same every run) "
              "validate — listed in aioqc/val_images.txt", flush=True)
    counts = Counter(c for _p, c in train_items)
    recipe = None
    aug = spec.get("augment") or {}
    if aug.get("recipe"):
        try:
            import aioqc_augment                       # copied next to this file by AIO QC
            recipe = aioqc_augment.ImageRecipe(aug["recipe"], float(aug.get("share", 1.0)))
            line = aioqc_augment.recipe_line(aioqc_augment.Augmenter(aug["recipe"]))
            print(line, flush=True)
            status.set(augment=line)
        except Exception as exc:
            print(f"WARNING: the augmentation recipe could not be used ({type(exc).__name__}: {exc})", flush=True)
            status.set(augment=f"not used: {type(exc).__name__}: {exc}")
    train_ds = FolderData(train_items, transforms(args, mean, std, True, size), recipe)
    eval_tf = transforms(args, mean, std, False, size)
    val_ds = FolderData(val_items, eval_tf)

    batch = int(_a(args, "batch")) if not isinstance(_a(args, "batch"), float) else 32
    if batch <= 0:
        batch = _auto_batch(net, device, size, len(train_items), amp, status)
    batch = max(1, min(batch, len(train_items)))
    workers = int(_a(args, "workers"))

    # ---------------------------------------------------------------- sampling (class balance / hard examples)
    bal = str(_a(args, "balance") or "none").lower()
    sampler, class_w, item_w = None, None, None
    if bal in ("sampler", "hard"):
        top = max(counts.values())
        cw = {c: min(top / n, 10.0) / top for c, n in counts.items()} if bal == "sampler" else {c: 1.0 for c in counts}
        w = [cw[c] for _p, c in train_items]
        if bal == "hard":
            if not start_trained and state is None:
                print("Hard examples need a trained model to start from (its best.pt) — sampling evenly instead",
                      flush=True)
            else:
                status.set(phase="finding the hard examples (the starting model on every training image)")
                ev = evaluate(net, _loader(FolderData(train_items, eval_tf), batch, workers, device=device), device,
                              nc, amp, keep=True)
                boost = float(_a(args, "hard_boost"))
                wrong = 0
                for i, pred, _conf, pt in ev["rows"]:
                    w[i] *= 1.0 + boost * (1.0 - pt)
                    wrong += pred != train_items[i][1]
                print(f"Hard examples: the starting model gets {wrong:,} of {len(train_items):,} training images "
                      f"wrong ({ev['top1']:.1%} right) — those are drawn up to {1 + boost:g}× as often", flush=True)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train_items), replacement=True)
        item_w = w
    elif bal == "loss":
        tot = sum(counts.values())
        class_w = torch.tensor([min(10.0, tot / (nc * max(counts.get(c, 0), 1))) for c in range(nc)],
                               device=device)
        class_w = class_w / class_w.mean()
    train_dl = _loader(train_ds, batch, workers, shuffle=True, sampler=sampler, drop_last=len(train_items) > batch,
                       device=device)
    val_dl = _loader(val_ds, max(batch, 32), workers, device=device)

    # ---------------------------------------------------------------- optimisation
    epochs = int(_a(args, "epochs"))
    iters = max(1, len(train_dl))
    lr = auto_lr(args, batch, bool(_a(args, "pretrained")) or start_trained or state is not None)
    if str(_a(args, "grad_ckpt")).lower() in ("on", "true", "1"):
        if hasattr(net, "set_grad_checkpointing"):
            net.set_grad_checkpointing(True)
            print("GPU memory saver on (gradient checkpointing)", flush=True)
        else:
            print("GPU memory saver: not available for this model", flush=True)
    opt, opt_name = _optimizer(net, info, args, lr, batch)
    for g in opt.param_groups:
        g["base_lr"] = lr * float(g.get("lr_scale", 1.0))
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") and \
        hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp)
    probe = int(_a(args, "freeze_epochs") or 0) >= int(_a(args, "epochs")) and \
        str(_a(args, "probe_features")).lower() == "cached"
    ema_decay = float(_a(args, "ema") or 0) if not probe else 0.0
    ema = EMA(net, ema_decay) if ema_decay > 0 else None
    start_epoch, best, best_epoch, step = 0, -1.0, 0, 0
    rows = []
    if state is not None:
        if state.get("optimizer") is not None:
            opt.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.module = state["ema"].float().to(device).eval()
            ema.updates = int(state.get("ema_updates", 0))
        start_epoch, best, best_epoch = int(state["epoch"]) + 1, float(state["best"]), int(state["best_epoch"])
        step = int(state.get("step", start_epoch * iters))
        rows = list(state.get("rows") or [])
    if ema is not None:
        ema.module.to(device)
    warm = int(float(_a(args, "warmup_epochs")) * iters)
    total = epochs * iters
    lrf, cos = float(_a(args, "lrf")), bool(_a(args, "cos_lr"))
    ls = float(_a(args, "label_smoothing") or 0)
    mixup, cutmix = float(_a(args, "mixup") or 0), float(_a(args, "cutmix") or 0)
    freeze = int(_a(args, "freeze_epochs") or 0)
    patience = int(_a(args, "patience") or 0)
    clip = float(_a(args, "grad_clip") or 0)
    limit = float(_a(args, "time") or 0) * 3600
    t0 = time.time()
    print(f"{info['lib']} {info['arch']} · {nc} classes · {len(train_items):,} train / {len(val_items):,} val images "
          f"· {size} px · batch {batch} · {opt_name} lr {lr:g} · {device}"
          + (f" · EMA {ema_decay}" if ema else "")
          + (" · linear probe (features once)" if probe else
             f" · head only for {min(freeze, epochs)} epoch(s)" if freeze else "")
          + (f" · balance: {bal}" if bal != "none" else ""), flush=True)
    status.set(state="training", phase="training", epochs=epochs, start_epoch=start_epoch, save_dir=str(run),
               train_images=len(train_items), val_images=len(val_items), batch_size=batch, device=str(device),
               names={i: n for i, n in enumerate(names)}, engine=f"{info['lib']} {info['arch']}", imgsz=size,
               classes_matched=remapped)

    stopped_by = ""
    epoch = start_epoch - 1
    if probe:                                            # a linear probe on features computed once
        best, best_epoch, rows, stopped_by = linear_probe(
            net, info, names, train_items, val_items, eval_tf, args, device, amp, batch, workers, status, ctrl,
            item_w, class_w, wdir, run, root, t0)
    for epoch in (range(start_epoch, epochs) if not probe else ()):
        head_only = epoch < freeze
        _set_trainable(net, info, head_only)
        net.train()
        if head_only:                                    # a linear probe: the backbone's BatchNorm stays as it is
            net.eval()
            head(net, info).train()
        status.set(phase="training" + (" the head only (backbone frozen)" if head_only else ""), epoch=epoch + 1,
                   batch=0, batches=iters)
        t_ep, loss_sum, seen, last_t = time.time(), 0.0, 0, 0.0
        for bi, (x, y) in enumerate(train_dl):
            f = lr_factor(step, total, warm, lrf, cos)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            soft = torch.nn.functional.one_hot(y, nc).float()
            x, soft = _mix(x, soft, mixup, cutmix, rng)
            if ls:
                soft = soft * (1 - ls) + ls / nc
            with torch.autocast(device_type=device.type, enabled=amp):
                out = net(x)
            loss = _soft_ce(out, soft, class_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(net)
            step += 1
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)
            now = time.time()
            if now - last_t > 1.0 or bi + 1 == iters:
                last_t = now
                extra = {}
                if device.type == "cuda":
                    extra["gpu_mem_gb"] = round(torch.cuda.memory_reserved(device) / 2 ** 30, 2)
                status.set(batch=bi + 1, batches=iters, loss={"train/loss": round(loss_sum / max(seen, 1), 5)},
                           **extra)
            if ctrl.get("stop"):
                break
        status.set(phase="validating")
        model_eval = ema.module if ema is not None else net
        ev = evaluate(model_eval, val_dl, device, nc, amp)
        t1 = ev["top1"]
        improved = t1 > best
        if improved:
            best, best_epoch = t1, epoch + 1
            save_best(wdir / "best.pt", model_eval, info, names, size,
                      {"top1": round(t1, 5), "top5": round(ev["top5"], 5), "epoch": epoch + 1, "data": str(root)})
        row = {"epoch": epoch + 1, "epochs": epochs, "loss": {"train/loss": round(loss_sum / max(seen, 1), 5)},
               "metrics": {"metrics/accuracy_top1": round(t1, 5), "metrics/accuracy_top5": round(ev["top5"], 5),
                           "val/loss": round(ev["loss"], 5)},
               "lr": {"lr/pg0": round(opt.param_groups[0]["lr"], 8)}, "time": round(time.time() - t_ep, 2),
               "fitness": round(t1, 5), "best_fitness": round(best, 5), "final": False, "t": time.time()}
        rows.append(row)
        status.event(row)
        _write_results_csv(run, rows)
        _atomic_save({"format": FORMAT, "net": copy.deepcopy(net).float().cpu(), "info": info, "args": args,
                      "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                      "ema": copy.deepcopy(ema.module).float().cpu() if ema is not None else None,
                      "ema_updates": ema.updates if ema is not None else 0, "epoch": epoch, "best": best,
                      "best_epoch": best_epoch, "step": step, "rows": rows}, wdir / "last.pt")
        status.set(metrics=row["metrics"], best_fitness=round(best, 5), epoch_time=row["time"], last_epoch=epoch + 1,
                   elapsed=round(time.time() - t0, 1), phase="training", best=str(wdir / "best.pt"),
                   last=str(wdir / "last.pt"), best_epoch=best_epoch)
        if ctrl.get("stop"):
            stopped_by = "stop"
            break
        if ctrl.get("pause"):
            status.set(state="paused", message="Paused after a saved epoch — Resume continues from it",
                       resumable=True, ended=time.time())
            release(0)
        if patience and epoch + 1 - best_epoch >= patience:
            stopped_by = "patience"
            print(f"Stopping: no better val accuracy for {patience} epochs (best: epoch {best_epoch}, "
                  f"top-1 {best:.4f})", flush=True)
            break
        if limit and time.time() - t0 > limit:
            stopped_by = "time"
            print("Stopping: the time limit is reached", flush=True)
            break

    # ---------------------------------------------------------------- the final check with best.pt
    status.set(phase="final check with best.pt")
    final = {}
    per = []
    if (wdir / "best.pt").exists():
        best_net, _i, _n, _ck = load(wdir / "best.pt")
        best_net = best_net.to(device)
        ev = evaluate(best_net, val_dl, device, nc, amp, keep=True)
        per = per_class(ev["cm"], names, counts)
        final = {"metrics/accuracy_top1": round(ev["top1"], 5), "metrics/accuracy_top5": round(ev["top5"], 5),
                 "val/loss": round(ev["loss"], 5)}
        _write_reports(run, names, val_items, ev)
        status.event({"epoch": best_epoch, "epochs": epochs, "loss": {}, "metrics": final, "lr": {}, "time": 0,
                      "fitness": final["metrics/accuracy_top1"], "best_fitness": round(best, 5), "final": True,
                      "t": time.time()})
        if bool(_a(args, "onnx")):
            status.set(phase="saving best.onnx (for use outside AIO QC)")
            try:
                seq = deploy(best_net.cpu(), info, names, size)
                export_onnx(seq, wdir / "best.onnx", size, names, info)
                print(f"Saved {wdir / 'best.onnx'}", flush=True)
            except Exception as exc:
                print(f"WARNING: best.onnx was not saved ({type(exc).__name__}: {exc}) — best.pt works everywhere "
                      "in AIO QC", flush=True)
    ext = spec.get("extend")
    if ext and stopped_by != "stop" and (wdir / "best.pt").exists():
        try:
            out = extend_compare(ext, wdir / "best.pt", args, device, status)
            (meta / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        except Exception as exc:
            print(f"WARNING: the before / after check did not run ({type(exc).__name__}: {exc})", flush=True)
    msg = {"stop": "Stopped early — the best epoch so far was checked and kept",
           "patience": f"Finished: no better for {patience} epochs (best: epoch {best_epoch})",
           "time": "Finished: the time limit was reached"}.get(stopped_by, "Finished")
    status.set(state="stopped" if stopped_by == "stop" else "finished", phase="done", per_class=per,
               metrics=final, best=str(wdir / "best.pt"), last=str(wdir / "last.pt"), save_dir=str(run),
               ended=time.time(), resumable=stopped_by == "stop", elapsed=round(time.time() - t0, 1), message=msg,
               onnx=str(wdir / "best.onnx") if (wdir / "best.onnx").exists() else None)


def _accuracy_by_name(path, rows, tree: Path, args, device, status, what):
    """Per-class accuracy (by class NAME) of the model in `path` on "image<TAB>class" rows."""
    net, info, names, _ck = load(path)
    net = net.to(device)
    size = int(info.get("imgsz") or 224)
    status.set(phase=f"before / after · {what}")
    items = [(str(tree / p), names.index(c) if c in names else -1) for p, c in rows]
    tf = transforms(args, info.get("mean") or IMAGENET_MEAN, info.get("std") or IMAGENET_STD, False, size)
    keep = [(p, max(c, 0)) for p, c in items]
    ev = evaluate(net, _loader(FolderData(keep, tf), 32, int(_a(args, "workers")), device=device), device,
                  len(names), keep=True)
    hit, tot = Counter(), Counter()
    for i, pred, _conf, _pt in ev["rows"]:
        c = rows[i][1]
        tot[c] += 1
        hit[c] += names[pred] == c
    return [{"name": c, "map": hit[c] / tot[c], "map50": hit[c] / tot[c], "mask_map": None} for c in sorted(tot)]


def extend_compare(ext, best, args, device, status):
    """Extend Model: the base model and the extended one, per class, on the old and the new validation images."""
    tree = Path(ext["tree"])
    out = {"before": {}, "after": {}}
    for split in ("val_old", "val_new"):
        f = tree / f"{split}.txt"
        if not f.is_file():
            continue
        rows = [ln.split("\t") for ln in f.read_text(encoding="utf-8").splitlines() if "\t" in ln]
        if not rows:
            continue
        out["after"][split] = _accuracy_by_name(best, rows, tree, args, device, status, f"the extended model on {split}")
        out["before"][split] = _accuracy_by_name(ext["base"], rows, tree, args, device, status,
                                                 f"the base model on {split}")
    return out


def validate(spec, status):
    args = dict(spec["args"])
    meta = Path(spec["_meta"])
    out = meta.parent
    status.set(state="starting", phase="loading the model")
    net, info, names, _ck = load(spec["model"])
    device = _device(args)
    net = net.to(device)
    size = snap(int(info.get("imgsz") or args.get("imgsz") or 224), patch_of(info.get("arch")))
    split = str(args.get("split") or "val")
    items = gather(Path(args["data"]), names, split, float(_a(args, "val_share")))
    if not items:
        raise RuntimeError(f"No {split} images for this model's classes in {args['data']}")
    status.set(state="validating", phase=f"validating on {split} ({len(items):,} images)")
    tf = transforms(args, info.get("mean") or IMAGENET_MEAN, info.get("std") or IMAGENET_STD, False, size)
    b = int(args.get("batch") or 32)
    ev = evaluate(net, _loader(FolderData(items, tf), b if b > 0 else 32, int(_a(args, "workers")), device=device),
                  device, len(names), device.type == "cuda", keep=True)
    _write_reports(out, names, items, ev)
    res = {"metrics/accuracy_top1": round(ev["top1"], 5), "metrics/accuracy_top5": round(ev["top5"], 5),
           "val/loss": round(ev["loss"], 5), "top1": round(ev["top1"], 5), "top5": round(ev["top5"], 5)}
    status.set(state="finished", phase="done", metrics=res, per_class=per_class(ev["cm"], names), save_dir=str(out),
               ended=time.time(), message=f"Validation finished: top-1 {ev['top1']:.1%} on {len(items):,} {split} "
                                          "images (val_predictions.csv lists every image)")


def run(spec, status, ctrl, release, meta: Path):
    spec = dict(spec, _meta=str(meta))
    if spec.get("mode") == "val":
        validate(spec, status)
    else:
        train(spec, status, ctrl, release)
