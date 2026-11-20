# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model training (Qt-free): check a YOLO dataset, plan a run, start it detached in the user's Python environment, and
read its progress back. The work itself is done by train_runner.py (Ultralytics) inside that environment.

    chk = check_dataset("…/dataset.yaml")             # counts, label format, issues (error / warn / info), fixes
    imgsz = suggest_imgsz(chk)                        # from the object sizes
    run = prepare_run(python, chk, settings)          # <project>/<name>/aioqc/spec.json + a resolved data.yaml
    launch(run)                                       # detached: the run survives the app closing
    status(run_dir), epochs(run_dir), log_tail(run_dir), control(run_dir, "stop" | "pause" | "kill")
    list_runs(project)                                # history, also runs made outside AIO QC (results.csv)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .appdirs import data_home
from AIO_QC.core.diag import ignored as _ignored

RUNNER = Path(__file__).with_name("train_runner.py")
IMG_EXT = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".pfm", ".avif", ".heic",
           ".heif", ".jp2"}
RUNNING = {"starting", "training", "validating"}
WIN = sys.platform == "win32"

# (family, sizes, det weights pattern, seg weights pattern)
FAMILIES = [
    ("YOLO26", "nsmlx", "yolo26{s}.pt", "yolo26{s}-seg.pt"),
    ("YOLO11", "nsmlx", "yolo11{s}.pt", "yolo11{s}-seg.pt"),
    ("YOLOv8", "nsmlx", "yolov8{s}.pt", "yolov8{s}-seg.pt"),
]
SIZE_WORD = {"n": "nano — fastest", "s": "small", "m": "medium", "l": "large", "x": "extra large — most accurate"}
PRESETS = {
    "quick": {"title": "Quick look", "size": "n", "epochs": 30, "patience": 10,
              "tip": "A first model in minutes: does the data learn at all? Which classes struggle?"},
    "balanced": {"title": "Balanced", "size": "s", "epochs": 100, "patience": 30,
                 "tip": "A good default: a small model, stops by itself when val stops improving"},
    "best": {"title": "Best accuracy", "size": "m", "epochs": 300, "patience": 60,
             "tip": "A bigger model, trained long — needs a GPU"},
}


# ------------------------------------------------------------------ classifiers of any architecture
# Trained by AIO QC's PyTorch trainer (core/cls_engine.py) in the same environment: timm when it is installed (every
# model here, pretrained weights from Hugging Face), else torchvision (those with a torchvision name).
# (key, title, group, timm model, torchvision model, image size, parameters in millions)
CLS_GROUPS = {"mobile": "Small & fast (edge, CPU)", "cnn": "CNNs", "vit": "Transformers",
              "found": "Foundation features (few labelled images)"}
CLS_ARCHS = [
    ("mobilenetv3_small", "MobileNetV3 Small", "mobile", "mobilenetv3_small_100.lamb_in1k", "mobilenet_v3_small", 224,
     2.5),
    ("mobilenetv3_large", "MobileNetV3 Large", "mobile", "mobilenetv3_large_100.ra_in1k", "mobilenet_v3_large", 224,
     5.5),
    ("mobilenetv4_small", "MobileNetV4 Conv Small", "mobile", "mobilenetv4_conv_small.e2400_r224_in1k", None, 224, 3.8),
    ("mobilenetv4_medium", "MobileNetV4 Conv Medium", "mobile", "mobilenetv4_conv_medium.e500_r256_in1k", None, 256,
     9.7),
    ("shufflenet_v2", "ShuffleNetV2 1.0", "mobile", None, "shufflenet_v2_x1_0", 224, 2.3),
    ("ghostnetv2", "GhostNetV2 1.0", "mobile", "ghostnetv2_100.in1k", None, 224, 6.2),
    ("efficientnet_b0", "EfficientNet-B0", "mobile", "efficientnet_b0.ra_in1k", "efficientnet_b0", 224, 5.3),
    ("convnextv2_atto", "ConvNeXt V2 Atto", "mobile", "convnextv2_atto.fcmae_ft_in1k", None, 224, 3.7),
    ("fastvit_t8", "FastViT T8", "mobile", "fastvit_t8.apple_in1k", None, 256, 4.0),
    ("repvit_m1_1", "RepViT M1.1", "mobile", "repvit_m1_1.dist_450e_in1k", None, 224, 8.8),
    ("efficientvit_b1", "EfficientViT B1", "mobile", "efficientvit_b1.r224_in1k", None, 224, 9.1),
    ("resnet18", "ResNet-18", "cnn", "resnet18.a1_in1k", "resnet18", 224, 11.7),
    ("resnet34", "ResNet-34", "cnn", "resnet34.a1_in1k", "resnet34", 224, 21.8),
    ("resnet50", "ResNet-50", "cnn", "resnet50.a1_in1k", "resnet50", 224, 25.6),
    ("resnet101", "ResNet-101", "cnn", "resnet101.a1h_in1k", "resnet101", 224, 44.5),
    ("resnext50", "ResNeXt-50 32×4d", "cnn", "resnext50_32x4d.a1h_in1k", "resnext50_32x4d", 224, 25.0),
    ("efficientnet_b2", "EfficientNet-B2", "cnn", "efficientnet_b2.ra_in1k", "efficientnet_b2", 256, 9.1),
    ("efficientnet_b3", "EfficientNet-B3", "cnn", "efficientnet_b3.ra2_in1k", "efficientnet_b3", 288, 12.2),
    ("efficientnetv2_s", "EfficientNetV2-S", "cnn", "tf_efficientnetv2_s.in21k_ft_in1k", "efficientnet_v2_s", 288,
     21.5),
    ("efficientnetv2_m", "EfficientNetV2-M", "cnn", "tf_efficientnetv2_m.in21k_ft_in1k", "efficientnet_v2_m", 384,
     54.1),
    ("regnety_016", "RegNetY-1.6GF", "cnn", "regnety_016.tv2_in1k", "regnet_y_1_6gf", 224, 11.2),
    ("regnety_032", "RegNetY-3.2GF", "cnn", "regnety_032.ra_in1k", "regnet_y_3_2gf", 224, 19.4),
    ("densenet121", "DenseNet-121", "cnn", "densenet121.ra_in1k", "densenet121", 224, 8.0),
    ("inception_v3", "Inception-v3", "cnn", "inception_v3.tv_in1k", None, 288, 23.8),
    ("convnext_nano", "ConvNeXt Nano", "cnn", "convnext_nano.in12k_ft_in1k", None, 224, 15.6),
    ("convnextv2_nano", "ConvNeXt V2 Nano", "cnn", "convnextv2_nano.fcmae_ft_in22k_in1k", None, 224, 15.6),
    ("convnext_tiny", "ConvNeXt Tiny", "cnn", "convnext_tiny.fb_in22k_ft_in1k", "convnext_tiny", 224, 28.6),
    ("convnextv2_tiny", "ConvNeXt V2 Tiny", "cnn", "convnextv2_tiny.fcmae_ft_in22k_in1k", None, 224, 28.6),
    ("convnext_small", "ConvNeXt Small", "cnn", "convnext_small.fb_in22k_ft_in1k", "convnext_small", 224, 50.2),
    ("vit_tiny", "ViT Tiny/16", "vit", "vit_tiny_patch16_224.augreg_in21k_ft_in1k", None, 224, 5.7),
    ("vit_small", "ViT Small/16", "vit", "vit_small_patch16_224.augreg_in21k_ft_in1k", None, 224, 22.1),
    ("vit_base", "ViT Base/16", "vit", "vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit_b_16", 224, 86.6),
    ("deit3_small", "DeiT III Small", "vit", "deit3_small_patch16_224.fb_in22k_ft_in1k", None, 224, 22.1),
    ("swin_tiny", "Swin Tiny", "vit", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "swin_t", 224, 28.3),
    ("tiny_vit_5m", "TinyViT 5M", "vit", "tiny_vit_5m_224.dist_in22k_ft_in1k", None, 224, 5.4),
    ("tiny_vit_21m", "TinyViT 21M", "vit", "tiny_vit_21m_224.dist_in22k_ft_in1k", None, 224, 21.2),
    ("eva02_tiny", "EVA-02 Tiny", "vit", "eva02_tiny_patch14_224.mim_in22k", None, 224, 5.5),
    ("eva02_small", "EVA-02 Small", "vit", "eva02_small_patch14_224.mim_in22k", None, 224, 21.6),
    # foundation features — self-supervised / image-text encoders whose features already separate most things: few
    # labelled images go a long way, a linear probe (the "Few labelled images" schedule) trains in minutes
    ("dinov3_small", "DINOv3 ViT-S/16", "found", "vit_small_patch16_dinov3.lvd1689m", None, 224, 21.6),
    ("dinov3_small_plus", "DINOv3 ViT-S+/16", "found", "vit_small_plus_patch16_dinov3.lvd1689m", None, 224, 28.7),
    ("dinov3_base", "DINOv3 ViT-B/16", "found", "vit_base_patch16_dinov3.lvd1689m", None, 224, 85.7),
    ("dinov3_large", "DINOv3 ViT-L/16", "found", "vit_large_patch16_dinov3.lvd1689m", None, 224, 300.2),
    ("dinov3_large_sat", "DINOv3 ViT-L/16 — satellite & aerial", "found", "vit_large_patch16_dinov3.sat493m", None, 224,
     300.2),
    ("dinov3_huge_plus", "DINOv3 ViT-H+/16", "found", "vit_huge_plus_patch16_dinov3.lvd1689m", None, 224, 840.6),
    ("dinov3_convnext_tiny", "DINOv3 ConvNeXt Tiny", "found", "convnext_tiny.dinov3_lvd1689m", None, 224, 27.8),
    ("dinov3_convnext_small", "DINOv3 ConvNeXt Small", "found", "convnext_small.dinov3_lvd1689m", None, 224, 49.5),
    ("dinov3_convnext_base", "DINOv3 ConvNeXt Base", "found", "convnext_base.dinov3_lvd1689m", None, 224, 87.6),
    ("dinov3_convnext_large", "DINOv3 ConvNeXt Large", "found", "convnext_large.dinov3_lvd1689m", None, 224, 196.2),
    ("dinov2_small", "DINOv2 ViT-S/14", "found", "vit_small_patch14_dinov2.lvd142m", None, 224, 22.1),
    ("dinov2_small_reg", "DINOv2 ViT-S/14 + registers", "found", "vit_small_patch14_reg4_dinov2.lvd142m", None, 224,
     22.1),
    ("dinov2_base", "DINOv2 ViT-B/14", "found", "vit_base_patch14_dinov2.lvd142m", None, 224, 86.6),
    ("dinov2_large", "DINOv2 ViT-L/14", "found", "vit_large_patch14_dinov2.lvd142m", None, 224, 304.4),
    ("dinov2_large_reg", "DINOv2 ViT-L/14 + registers", "found", "vit_large_patch14_reg4_dinov2.lvd142m", None, 224,
     304.4),
    ("siglip2_base", "SigLIP 2 ViT-B/16", "found", "vit_base_patch16_siglip_224.v2_webli", None, 224, 92.9),
    ("siglip2_so400m", "SigLIP 2 So400m/14", "found", "vit_so400m_patch14_siglip_224.v2_webli", None, 224, 428.2),
    ("clip_base", "CLIP ViT-B/16 (OpenAI)", "found", "vit_base_patch16_clip_224.openai", None, 224, 86.2),
    ("clip_large", "CLIP ViT-L/14 (OpenAI)", "found", "vit_large_patch14_clip_224.openai", None, 224, 304.0),
    ("pe_core_small", "Perception Encoder S/16", "found", "vit_pe_core_small_patch16_384.fb", None, 384, 22.3),
    ("pe_core_base", "Perception Encoder B/16", "found", "vit_pe_core_base_patch16_224.fb", None, 224, 86.4),
    ("pe_core_large", "Perception Encoder L/14", "found", "vit_pe_core_large_patch14_336.fb", None, 224, 317.0),
    ("aimv2_large", "AIMv2 L/14", "found", "aimv2_large_patch14_224.apple_pt", None, 224, 309.2),
    ("aimv2_huge", "AIMv2 H/14", "found", "aimv2_huge_patch14_224.apple_pt", None, 224, 680.9),
    # NVIDIA's C-RADIO: one encoder distilled from DINOv2 / v3, SigLIP, SAM … ("AM-RADIO")
    ("radio_v4_so400m", "NVIDIA C-RADIOv4 SO400M", "found", "vit_so400m_patch16_radio_v4.c_radio", None, 256, 412.7),
    ("radio_v4_huge", "NVIDIA C-RADIOv4 H", "found", "vit_huge_patch16_radio_v4.c_radio", None, 256, 631.0),
    ("radio_v3_base", "NVIDIA C-RADIOv3 B", "found", "vit_base_patch16_radio_v3.c_radio", None, 256, 85.9),
    ("radio_v3_large", "NVIDIA C-RADIOv3 L", "found", "vit_large_patch16_radio_v3.c_radio", None, 256, 303.4),
    ("radio_v3_huge", "NVIDIA C-RADIOv3 H", "found", "vit_huge_patch16_radio_v3.c_radio", None, 256, 631.0),
    # Google's TIPSv2: image-text pretraining with spatial awareness
    ("tipsv2_base", "TIPSv2 B/14 (Google)", "found", "vit_base_patch14_reg1_tipsv2.webli", None, 224, 85.7),
    ("tipsv2_large", "TIPSv2 L/14 (Google)", "found", "vit_large_patch14_reg1_tipsv2.webli", None, 224, 303.2),
    ("tipsv2_so400m", "TIPSv2 So400m/14 (Google)", "found", "vit_so400m_patch14_reg1_tipsv2.webli", None, 224, 412.5),
]
# the first timm release with a family (by key prefix): older ones are offered the update
TIMM_MIN = {"dinov3": "1.0.20", "pe_core": "1.0.16", "siglip2": "1.0.15", "aimv2": "1.0.13", "radio": "1.0.31",
            "tipsv2": "1.0.28"}
# weights under their own terms (the rest: Apache-2.0 / BSD / MIT, as timm lists them)
ARCH_LICENCE = {"dinov3": "the DINOv3 License (Meta) — read its terms before commercial use",
                "aimv2": "Apple's sample-code licence (ASCL)",
                "radio": "the NVIDIA Open Model License (commercial use allowed — read its terms)"}
CLS_ARCH = {a[0]: a for a in CLS_ARCHS}


def is_torch_model(model) -> bool:
    """Trained by the PyTorch classifier trainer: an architecture (arch: / timm: / tv:) or a model it saved (a
    best.aioqc.json beside the .pt)."""
    m = str(model or "")
    return m.startswith(("arch:", "timm:", "tv:")) or checkpoint_info(m) is not None


def checkpoint_info(path) -> dict | None:
    """What the PyTorch classifier trainer wrote beside its model (weights/best.aioqc.json): architecture, classes,
    image size, score. None for anything else (a YOLO .pt)."""
    m = str(path or "")
    if not m.lower().endswith((".pt", ".pth")):
        return None
    p = Path(m)
    for side in (p.with_name(p.stem + ".aioqc.json"), p.with_name("best.aioqc.json") if p.name == "last.pt" else None):
        if side is not None and side.is_file():
            try:
                return json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {"engine": "torch"}
    return None


def _ver(v) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", str(v or ""))[:3]) or (0,)


def timm_min(model: str) -> str | None:
    """The timm release an architecture needs (None: any)."""
    m = str(model or "")
    key = m[5:] if m.startswith("arch:") else next((a[0] for a in CLS_ARCHS if m == f"timm:{a[3]}"), "")
    return next((v for p, v in TIMM_MIN.items() if key.startswith(p)), None)


def arch_licence(model: str) -> str | None:
    m = str(model or "")
    key = m[5:] if m.startswith("arch:") else next((a[0] for a in CLS_ARCHS if m == f"timm:{a[3]}"), "")
    return next((v for p, v in ARCH_LICENCE.items() if key.startswith(p)), None)


def timm_update(model: str, env: dict | None) -> str | None:
    """"timm>=X" when the environment's timm is older than the architecture needs (None: fine, or unknown)."""
    need = timm_min(model)
    have = (env or {}).get("timm")
    if need and have and _ver(have) < _ver(need):
        return f"timm>={need}"
    return None


def resolve_arch(model: str, env: dict | None) -> tuple[str, str]:
    """"arch:<key>" -> ("timm:<name>" / "tv:<name>", "") — timm when the environment has it (or nothing is known),
    else torchvision when the architecture is there; ("", why) when it can't be trained here."""
    m = str(model or "")
    if not m.startswith("arch:"):
        return m, ""
    a = CLS_ARCH.get(m[5:])
    if a is None:
        return "", f"unknown architecture {m[5:]}"
    has_timm = (env or {}).get("timm") if env and "timm" in env else True
    if a[3] and has_timm:
        upd = timm_update(m, env)
        if upd and not a[4]:
            return "", f"{a[1]} needs timm {upd[6:]} or newer (this environment has {env.get('timm')}) — Update timm"
        if not upd:
            return f"timm:{a[3]}", ""
    if a[4]:
        return f"tv:{a[4]}", ""
    return "", f"{a[1]} needs timm in the training environment (Install timm)"


def arch_title(model: str) -> str:
    m = str(model or "")
    if m.startswith("arch:") and m[5:] in CLS_ARCH:
        return CLS_ARCH[m[5:]][1]
    for a in CLS_ARCHS:
        if m in (f"timm:{a[3]}", f"tv:{a[4]}"):
            return a[1]
    return m.split(":", 1)[-1]


def arch_imgsz(model: str) -> int | None:
    m = str(model or "")
    a = CLS_ARCH.get(m[5:]) if m.startswith("arch:") else next(
        (x for x in CLS_ARCHS if m in (f"timm:{x[3]}", f"tv:{x[4]}")), None)
    if a is not None:
        return int(a[5])
    info = checkpoint_info(m)
    return int(info["imgsz"]) if info and info.get("imgsz") else None


def torch_keys() -> set:
    from .cls_engine import KEYS
    return set(KEYS)


# ------------------------------------------------------------------ augmentation & learning-rate schedule
# (key, label, group, kind, lo, hi, step, decimals, suffix, Ultralytics default, tip, tasks or None)
# tasks: "det" / "seg" (Ultralytics detection / segmentation), "cls" (a YOLO classifier — Ultralytics), "clst" (a
# classifier of any other architecture — AIO QC's PyTorch trainer, core/cls_engine.py). None: every task.
# Groups: "lr" schedule, "aug" augmentation, "ft" fine-tuning (classifiers).
DET = ("det", "seg")
CLS = ("cls", "clst")
HYP = [
    ("lrf", "Final learning rate", "lr", "float", 0.0001, 1.0, 0.01, 4, " × start", 0.01,
     "Learning-rate decay: by the last epoch the rate has fallen to start × this (0.01: to 1 %)", None),
    ("cos_lr", "Decay curve", "lr", "choice", None, None, None, None, "", False,
     "How the rate falls from start to final: a straight line, or a cosine (slower at first, smoother at the end — "
     "often a little better on long runs)", None),
    ("warmup_epochs", "Warm-up", "lr", "float", 0.0, 20.0, 0.5, 1, " epochs", 3.0,
     "The first epochs ramp the learning rate up from near zero (stabilises the start; short for fine-tuning)", None),
    ("momentum", "Momentum", "lr", "float", 0.5, 0.999, 0.001, 3, "", 0.937,
     "SGD momentum / Adam beta1 — used with a chosen optimizer (auto picks its own)", None),
    ("weight_decay", "Weight decay", "lr", "float", 0.0, 0.1, 0.0001, 5, "", 0.0005,
     "L2 regularisation: higher fights over-fitting on small datasets (AdamW on a transformer: about 0.05)", None),
    ("freeze", "Freeze first", "lr", "int", 0, 100, 1, 0, " layers", None,
     "Fine-tuning: the first N layers keep their weights (10 ≈ the backbone of YOLOv8 / YOLO11) — faster, needs "
     "less data, keeps what the model already knows. 0: train every layer", ("det", "seg", "cls")),
    ("fliplr", "Mirror left–right", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.5,
     "Probability an image is mirrored. 0 for text — plates, signs, digits read backwards are not real", None),
    ("flipud", "Upside-down", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.0,
     "Probability an image is turned upside-down — only for top-down views (aerial, microscopy, conveyor)", None),
    ("mosaic", "Mosaic", "aug", "float", 0.0, 1.0, 0.05, 2, "", 1.0,
     "Probability of four images tiled into one: more objects, sizes and contexts per batch", DET),
    ("mixup", "MixUp", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.0,
     "Probability two images are blended (a classifier learns the blend of their classes): helps small datasets "
     "and over-fitting", ("det", "seg", "clst")),
    ("cutmix", "CutMix", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.0,
     "Probability a patch of another image is pasted in (occlusion robustness; a classifier learns both classes "
     "by their share)", ("det", "seg", "clst")),
    ("copy_paste", "Copy-paste objects", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.0,
     "Segmentation: share of objects copied into other images (more instances of rare classes)", ("seg",)),
    ("auto_augment", "Auto-augment policy", "aug", "pick", None, None, None, None, "", "randaugment",
     "A classifier's ready-made mix of colour, contrast, sharpness and small geometric changes, a different one "
     "per image. RandAugment: a strong default · AutoAugment: the policy learned on ImageNet · AugMix: blends "
     "several augmented copies (robust to blur / noise) · TrivialAugment: one random change, no tuning · None: "
     "the colour settings below instead", CLS),
    ("erasing", "Random erasing", "aug", "float", 0.0, 0.9, 0.05, 2, "", 0.4,
     "Classifiers: probability a random rectangle of the image is blanked out — the model learns from more than one "
     "part of the object", CLS),
    ("degrees", "Rotation", "aug", "float", 0.0, 180.0, 1.0, 1, " °", 0.0,
     "Random rotation up to ± this", ("det", "seg", "clst")),
    ("translate", "Shift", "aug", "float", 0.0, 0.9, 0.05, 2, "", 0.1,
     "Random shift up to ± this share of the image", DET),
    ("scale", "Zoom", "aug", "float", 0.0, 0.95, 0.05, 2, "", 0.5,
     "Random zoom in / out up to ± this share (objects of more sizes). Classifiers: a random crop keeping at least "
     "1 − this of the image (0.5: half or more) — lower it for small crops where the whole object matters", None),
    ("shear", "Shear", "aug", "float", 0.0, 45.0, 0.5, 1, " °", 0.0, "Random shear up to ± this", DET),
    ("perspective", "Perspective", "aug", "float", 0.0, 0.001, 0.0001, 4, "", 0.0,
     "Random perspective warp (0 … 0.001)", DET),
    ("hsv_h", "Hue", "aug", "float", 0.0, 1.0, 0.005, 3, "", 0.015,
     "Colour shift (a share of the colour wheel) — keep small when colour is what tells classes apart. Classifiers: "
     "used when the auto-augment policy is None", None),
    ("hsv_s", "Saturation", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.7, "Saturation change up to ± this share", None),
    ("hsv_v", "Brightness", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.4,
     "Brightness change up to ± this share (day / night, exposure)", None),
    ("bgr", "Swap colour channels", "aug", "float", 0.0, 1.0, 0.05, 2, "", 0.0,
     "Probability RGB ↔ BGR are swapped", DET),
    # ---- fine-tuning a classifier
    ("freeze_epochs", "Head only for the first", "ft", "int", 0, 1000, 1, 0, " epochs", 0,
     "A linear probe first: only the new classifier layer learns while the pretrained backbone stays as it is, then "
     "every layer trains. Protects the pretrained features from the random new head's first big updates. As many "
     "as the epochs: the head only, all the way (fastest; few images)", ("clst",)),
    ("layer_decay", "Layer-wise learning-rate decay", "ft", "float", 0.5, 1.0, 0.05, 2, "", 1.0,
     "Earlier layers learn slower than later ones (each block × this): the general features from pretraining "
     "change least. 0.65–0.8 for ViT / ConvNeXt / EfficientNet fine-tuning; 1: off. timm models only", ("clst",)),
    ("label_smoothing", "Label smoothing", "ft", "float", 0.0, 0.5, 0.05, 2, "", 0.1,
     "Targets of 0.9 / 0.1 instead of 1 / 0: a model less over-confident on the training images, and steadier "
     "with a few mislabelled ones", ("clst",)),
    ("ema", "Weight averaging (EMA)", "ft", "float", 0.0, 0.99999, 0.0001, 5, "", 0.9998,
     "Validate and keep a moving average of the weights (0.9998), not the last step's: smoother, usually a little "
     "more accurate. 0: off", ("clst",)),
    ("dropout", "Dropout before the classifier", "ft", "float", 0.0, 0.9, 0.05, 2, "", 0.0,
     "Randomly drops features before the last layer while training — against over-fitting on small datasets",
     CLS),
    ("drop_path", "Stochastic depth", "ft", "float", 0.0, 0.5, 0.05, 2, "", 0.0,
     "Randomly skips whole residual blocks while training (0.1 for ViT / ConvNeXt fine-tuning) — timm models "
     "only", ("clst",)),
    ("probe_features", "Head-only epochs", "ft", "pick", None, None, None, None, "", "cached",
     "When only the head learns all the way (Head only for the first ≥ the epochs — a linear probe): Features once "
     "runs every image through the backbone once (plus a mirrored copy) and then trains the head on those features "
     "— seconds per run, even for ViT-L on a CPU; With augmentation runs the backbone every epoch (slower, more "
     "variety for larger datasets)", ("clst",)),
    ("grad_ckpt", "GPU memory saver", "ft", "pick", None, None, None, None, "", "off",
     "Gradient checkpointing: recomputes activations instead of keeping them — fine-tuning a large model (ViT-L, "
     "ConvNeXt Large …) fits in less GPU memory, about a quarter slower. timm models only", ("clst",)),
    ("balance", "Class balance", "ft", "pick", None, None, None, None, "", "none",
     "Rare classes: Sampler draws them more often (up to 10×); Loss weights their mistakes more. Hard examples "
     "(starting from your trained model): images it gets wrong or is unsure of are drawn more often — the next "
     "round of improving", ("clst",)),
]
PICKS = {
    "auto_augment": [("randaugment", "RandAugment"), ("autoaugment", "AutoAugment (ImageNet)"), ("augmix", "AugMix"),
                     ("trivialaugment", "TrivialAugment"), ("none", "None — colour settings below")],
    "probe_features": [("cached", "Features once — fast (no augmentation)"),
                       ("augmented", "With augmentation — the backbone every epoch")],
    "grad_ckpt": [("off", "Off"), ("on", "On — less memory, ~25 % slower")],
    "balance": [("none", "As they are"), ("sampler", "Sampler — rare classes drawn more often"),
                ("loss", "Loss — rare classes' mistakes weigh more"),
                ("hard", "Hard examples — what the starting model gets wrong")],
}
HYP_KEYS = [h[0] for h in HYP]
HYP_DEFAULT = {h[0]: h[9] for h in HYP}
HYP_ROW = {h[0]: h for h in HYP}

# Presets: "values" for every task; "clst" (and "cls") values on top for classifiers. "tasks": only offered there.
AUG_PRESETS = {
    "default": {"title": "Default", "values": {},
                "tip": "Detection: mosaic, mirror, zoom, shift and colour (Ultralytics' defaults). Classifiers: a random "
                       "crop, mirror, RandAugment and random erasing — good for most datasets"},
    "light": {"title": "Light", "values": {"mosaic": 0.5, "scale": 0.3, "translate": 0.05, "hsv_s": 0.4,
                                          "hsv_v": 0.25},
              "cls": {"scale": 0.2, "erasing": 0.1, "auto_augment": "trivialaugment", "hsv_s": 0.2, "hsv_v": 0.2},
              "tip": "Gentler changes: fine-tuning, large datasets, images that all look alike (fixed cameras, tight "
                     "crops)"},
    "strong": {"title": "Strong", "values": {"mixup": 0.15, "copy_paste": 0.1, "degrees": 5.0, "scale": 0.7,
                                            "translate": 0.15, "hsv_v": 0.5},
               "cls": {"mixup": 0.2, "cutmix": 0.2, "erasing": 0.5, "degrees": 10.0, "scale": 0.7,
                       "auto_augment": "randaugment"},
               "tip": "More variety: small datasets that over-fit (train score far above val)"},
    "off": {"title": "Off", "values": {"mosaic": 0.0, "fliplr": 0.0, "translate": 0.0, "scale": 0.0, "hsv_h": 0.0,
                                      "hsv_s": 0.0, "hsv_v": 0.0, "auto_augment": "none", "erasing": 0.0},
            "tip": "No augmentation: a baseline, or checking that the model can learn the data at all"},
}
LR_PRESETS = {
    "default": {"title": "Default", "values": {"optimizer": "auto", "lr0": 0.0},
                "tip": "Optimizer and learning rate chosen for you (YOLO: from the run length; other classifiers: "
                       "AdamW, the rate from the batch size); 3 warm-up epochs"},
    "finetune": {"title": "Fine-tuning", "values": {"optimizer": "AdamW", "lr0": 0.0005, "lrf": 0.1,
                                                   "warmup_epochs": 1.0, "cos_lr": True, "freeze": 10},
                 "clst": {"lr0": 0.0002, "lrf": 0.05, "freeze_epochs": 2, "layer_decay": 0.75,
                          "label_smoothing": 0.1, "ema": 0.9998, "drop_path": 0.1, "weight_decay": 0.05},
                 "tip": "From a trained (or pretrained) model: a small learning rate, 1 warm-up epoch, cosine decay. "
                        "YOLO: the backbone (first 10 layers) frozen. Other classifiers: the head alone for 2 epochs, "
                        "then every layer with layer-wise decay (earlier layers slower), EMA, stochastic depth"},
    "improve": {"title": "Improve my model (hard examples)", "tasks": ("clst",),
                "values": {"optimizer": "AdamW", "lr0": 0.0001, "lrf": 0.1, "warmup_epochs": 0.5, "cos_lr": True},
                "clst": {"freeze_epochs": 0, "balance": "hard", "ema": 0.9998, "layer_decay": 1.0},
                "tip": "Start from your trained model (its best.pt) after fixing labels or adding images: a gentle "
                       "learning rate, and the images it gets wrong are drawn more often"},
    "few": {"title": "Few labelled images (linear probe)", "tasks": ("clst",),
            "values": {"optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "warmup_epochs": 0.0, "cos_lr": True},
            "clst": {"freeze_epochs": 1000, "probe_features": "cached", "label_smoothing": 0.1, "ema": 0.0,
                     "balance": "sampler", "weight_decay": 0.01, "layer_decay": 1.0, "drop_path": 0.0},
            "tip": "A handful to a few hundred images per class, with a foundation model (DINOv3, DINOv2, SigLIP 2, "
                   "CLIP, Perception Encoder …): the backbone stays as it is, every image goes through it once, and "
                   "only the classifier on top learns — seconds to minutes, little over-fitting, rare classes "
                   "drawn as often as common ones. More images later: Fine-tuning from this model"},
    "long": {"title": "Long run (cosine)", "values": {"cos_lr": True},
             "tip": "Cosine decay — usually a little better for runs of 100+ epochs"},
}
# what the PyTorch classifier trainer starts from (Ultralytics' defaults are for YOLO with SGD)
CLST_DEFAULT = {"weight_decay": 0.05, "momentum": 0.9, "hsv_s": 0.4, "erasing": 0.25, "cos_lr": True}


def preset_values(p: dict, task: str) -> dict:
    v = dict(p.get("values") or {})
    if task in CLS:
        v.update(p.get("cls") or {})
    if task == "clst":
        v.update(p.get("clst") or {})
    return v


TEXT_WORDS = re.compile(r"plate|text|digit|char|letter|number|ocr|lpr|licen[cs]e|sign|word|label_?id|serial",
                        re.IGNORECASE)


def text_like(names) -> bool:
    """Do the class names suggest text in the images (plates, characters, digits)? Then mirroring is wrong."""
    names = [str(n) for n in (names or [])]
    if any(TEXT_WORDS.search(n) for n in names):
        return True
    singles = [n for n in names if len(n.strip()) == 1 and n.strip().isalnum()]
    return len(singles) >= 5                              # 0-9 / A-Z character classes


def hyp_defaults(env: dict | None, task: str = "det") -> dict:
    """Ultralytics' defaults — the environment's own when it reported them (they move between versions). The
    PyTorch classifier trainer ("clst") has its own."""
    d = dict(HYP_DEFAULT)
    if task == "clst":
        d.update(CLST_DEFAULT)
        return d
    cfg = (env or {}).get("cfg") or {}
    for k in HYP_KEYS:
        if k in cfg:
            d[k] = cfg[k]
    return d


def supported_keys(env: dict | None) -> set | None:
    """The settings the environment's Ultralytics knows (None: not reported — an older check)."""
    keys = (env or {}).get("cfg_keys")
    return set(keys) if keys else None


def unknown_settings(keys, env: dict | None) -> list:
    known = supported_keys(env)
    return [] if known is None else [k for k in keys if k not in known]


def weights_name(family: str, size: str, task: str) -> str:
    for fam, _sizes, det, seg in FAMILIES:
        if fam == family:
            if task == "cls":
                return det.replace(".pt", "-cls.pt").format(s=size)
            return (seg if task == "seg" else det).format(s=size)
    raise KeyError(family)


def model_task(model: str) -> str | None:
    """'det' / 'seg' / 'cls' from a weights or yaml name (None: unknown, e.g. a custom .pt)."""
    if is_torch_model(model):
        return "cls"
    stem = Path(model).stem.lower()
    if stem.endswith("-seg") or "-seg" in stem:
        return "seg"
    if stem.endswith("-cls") or "-cls" in stem:
        return "cls"
    if re.match(r"^(yolo(v\d+|\d+)[nsmlx]?|rtdetr.*)$", stem):
        return "det"
    return None


def weights_dir() -> Path:
    """Where downloaded weights are kept (Ultralytics downloads into the working directory)."""
    d = data_home() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ====================================================================================================== dataset
@dataclass
class DataCheck:
    yaml: Path
    root: Path | None = None                  # the dataset's "path", resolved
    data: dict = field(default_factory=dict)
    names: list = field(default_factory=list)
    splits: dict = field(default_factory=dict)        # split -> {"images", "labels", "empty", "missing", "dirs"}
    instances: dict = field(default_factory=dict)     # split -> Counter(class id)
    fmt: Counter = field(default_factory=Counter)     # "box" / "polygon" lines
    bad_lines: int = 0
    bad_ids: Counter = field(default_factory=Counter)
    other_formats: Counter = field(default_factory=Counter)
    sizes: list = field(default_factory=list)         # (W, H) of a sample of train images
    objects: list = field(default_factory=list)       # (w, h) normalised, a sample of train objects
    issues: list = field(default_factory=list)        # (level, text)
    fixed_path: str | None = None             # "path" was unusable; the yaml's own folder is used
    cls_folder: bool = False                  # a classification dataset: <root>/{train,val,test}/<class>/<image>
    flat: bool = False                        # … or <root>/<class>/<image>: the PyTorch trainer splits it itself

    @property
    def ok(self) -> bool:
        return not any(lv == "error" for lv, _t in self.issues)

    @property
    def task(self) -> str:
        if self.cls_folder:
            return "cls"
        return "seg" if self.fmt["polygon"] > self.fmt["box"] else "det"

    def n(self, split: str, key: str = "images") -> int:
        return self.splits.get(split, {}).get(key, 0)


def _load_yaml(p: Path) -> dict:
    """PyYAML when installed (it is optional for AIO QC), else the small reader Testing uses."""
    from .testing.dataset import load_yaml
    d = load_yaml(p)
    if not isinstance(d, dict):
        raise ValueError("not a mapping")
    for k in ("train", "val", "test"):                 # a block list read without PyYAML: {0: …, 1: …}
        v = d.get(k)
        if isinstance(v, dict) and v and all(isinstance(i, int) for i in v):
            d[k] = [v[i] for i in sorted(v)]
    return d


def _scalar(v: str):
    """'0.5' / '10' / 'true' / 'null' / '[1, 2]' / text -> a Python value (YAML-like, no PyYAML needed)."""
    import ast
    low = v.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return ast.literal_eval(v.strip())
    except (ValueError, SyntaxError):
        return v.strip().strip("'\"")


def _names(d: dict) -> list:
    n = d.get("names")
    if isinstance(n, dict):
        try:
            return [str(n[k]) for k in sorted(n, key=lambda k: int(k))]
        except (TypeError, ValueError):
            return [str(v) for v in n.values()]
    if isinstance(n, (list, tuple)):
        return [str(x) for x in n]
    if isinstance(d.get("nc"), int):
        return [f"class{i}" for i in range(d["nc"])]
    return []


def _label_path(img: Path) -> Path:
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in s:
        s = sb.join(s.rsplit(sa, 1))
    return Path(s).with_suffix(".txt")


def _images_of(entry, root: Path) -> tuple[list[Path], list[str]]:
    """A split entry (folder / .txt list / list of them) -> images, problems."""
    out, probs = [], []
    for e in entry if isinstance(entry, (list, tuple)) else [entry]:
        p = Path(str(e))
        p = p if p.is_absolute() else (root / p)
        if p.is_dir():
            for dp, _dn, fn in os.walk(p):
                for f in fn:
                    if Path(f).suffix.lower() in IMG_EXT:
                        out.append(Path(dp) / f)
        elif p.is_file() and p.suffix.lower() == ".txt":
            base = p.parent
            for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip()
                if ln:
                    q = Path(ln)
                    out.append(q if q.is_absolute() else (base / q).resolve())
        else:
            probs.append(str(p))
    return out, probs


FLAT_VAL = 0.15                               # a flat class-folder dataset: the share of each class held out


def _class_dirs(base: Path) -> list:
    from .issues_folder import is_issues_dir
    return sorted((d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")
                   and d.name.lower() not in ("runs", "labels", "images") and not is_issues_dir(d)),
                  key=lambda d: d.name)


def _count_images(folder: Path, cancel=None) -> int:
    from .dataset import IMAGE_EXTENSIONS
    n = 0
    for _dp, dn, fns in os.walk(folder):
        dn[:] = [d for d in dn if not d.startswith(".")]
        if cancel is not None and cancel.is_set():
            return n
        n += sum(1 for f in fns if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
    return n


def check_cls_folder(root: Path, cancel=None) -> DataCheck:
    """A classification dataset: <root>/train/<class>/… and val/ (or valid/ / test/) — class ids are the class
    folders in name order (as Ultralytics reads them). Or a flat folder of class folders (<root>/<class>/…): the
    PyTorch trainer holds out FLAT_VAL of each class for validation itself (chk.flat); YOLO classifiers need the
    split made first."""
    from .dataset import class_folder_splits
    root = Path(root)
    chk = DataCheck(yaml=root, root=root, cls_folder=True)
    splits = dict(class_folder_splits(root))
    if "train" not in splits:
        dirs = [d for d in _class_dirs(root) if d.name.lower() not in ("train", "val", "valid", "test")]
        per = Counter()
        for d in dirs:
            if cancel is not None and cancel.is_set():
                return chk
            n = _count_images(d, cancel)
            if n:
                per[d.name] = n
        if len(per) >= 2 and not splits:
            chk.flat = True
            chk.names = sorted(per)
            nv = {k: (max(1, int(round(v * FLAT_VAL))) if v >= 2 else 0) for k, v in per.items()}
            chk.splits["train"] = {"images": sum(per.values()) - sum(nv.values()), "labels": 0, "empty": 0,
                                   "missing": 0, "dirs": [str(root)]}
            chk.splits["val"] = {"images": sum(nv.values()), "labels": 0, "empty": 0, "missing": 0,
                                 "dirs": [str(root)]}
            chk.instances["train"] = Counter({chk.names.index(k): v - nv[k] for k, v in per.items()})
            chk.instances["val"] = Counter({chk.names.index(k): nv[k] for k, v in per.items()})
            chk.issues.append(("info", f"No train/ and val/ folders: {FLAT_VAL:.0%} of each class validates (the "
                                       "same images every run, kept when images are added). YOLO classifiers need a "
                                       "split first: Convert & Split ▸ Class folders."))
            few = [k for k, v in per.items() if v < 10]
            if few:
                chk.issues.append(("warn", f"Fewer than 10 images: {', '.join(few[:8])}"))
            return chk
        chk.issues.append(("error", "No train/ folder of class folders (Ultralytics needs <dataset>/train/<class>/… "
                                    "and val/). Convert & Split ▸ Class folders makes one with a balanced split."))
        return chk
    if "val" not in splits and "test" not in splits:
        chk.issues.append(("error", "No val/ (or test/) folder — Ultralytics validates on it. Convert & Split makes "
                                    "one."))
    names = [d.name for d in _class_dirs(splits["train"])]
    chk.names = names
    for sp, d in splits.items():
        per = Counter()
        for c in _class_dirs(d):
            if cancel is not None and cancel.is_set():
                return chk
            per[c.name] = _count_images(c, cancel)
        chk.splits[sp] = {"images": sum(per.values()), "labels": sum(per.values()), "empty": 0, "missing": 0,
                          "dirs": [str(d)]}
        chk.instances[sp] = Counter({names.index(k): v for k, v in per.items() if k in names})
        extra = sorted(set(per) - set(names))
        if extra:
            chk.issues.append(("error", f"{sp}/ has class folders train/ does not: {', '.join(extra[:8])} — "
                                        "Ultralytics reads the classes from train/"))
        empty = [k for k in names if not per.get(k)]
        if empty and sp != "train":
            chk.issues.append(("warn", f"No {sp} images of: {', '.join(empty[:8])}"))
        elif empty:
            chk.issues.append(("error", f"Empty class folders in train/: {', '.join(empty[:8])}"))
    if len(names) < 2:
        chk.issues.append(("error", "A classifier needs at least 2 classes."))
    few = [k for k in names if chk.instances.get("train", {}).get(names.index(k), 0) < 10]
    if few:
        chk.issues.append(("warn", f"Fewer than 10 training images: {', '.join(few[:8])}"))
    return chk


def check_dataset(yaml_path, progress=None, cancel=None, sample_sizes: int = 300) -> DataCheck:
    """Everything Ultralytics will trip over, before a run starts — and the numbers the settings are chosen from."""
    yp = Path(yaml_path)
    if yp.is_dir():                                   # a folder of class folders: a classification dataset
        return check_cls_folder(yp, cancel)
    chk = DataCheck(yaml=yp)
    try:
        d = _load_yaml(yp)
    except Exception as exc:
        _ignored(exc)
        chk.issues.append(("error", f"Could not read {yp.name}: {exc}"))
        return chk
    chk.data = d
    chk.names = _names(d)
    if not chk.names:
        chk.issues.append(("error", "No class names (names:) in the dataset file."))
    if isinstance(d.get("nc"), int) and chk.names and d["nc"] != len(chk.names):
        chk.issues.append(("warn", f"nc is {d['nc']} but {len(chk.names)} names are listed — the names win."))
    # the dataset root, as Ultralytics resolves it (and a fix when it points somewhere that is gone)
    raw = d.get("path")
    root = Path(str(raw)) if raw else yp.parent
    if raw and not root.is_absolute():
        root = (yp.parent / root).resolve()
    if not root.exists() or not any((root / str(d.get(s, ""))).exists() for s in ("train", "val") if d.get(s)):
        if any((yp.parent / str(d.get(s, ""))).exists() for s in ("train", "val") if isinstance(d.get(s), str)):
            chk.fixed_path = str(yp.parent)
            chk.issues.append(("warn", f"“path: {raw}” does not exist here (made on another computer?) — the "
                                       f"dataset file's own folder is used instead."))
            root = yp.parent
    chk.root = root
    total = 0
    for sp in ("train", "val", "test"):
        if not d.get(sp):
            continue
        imgs, probs = _images_of(d[sp], root)
        for p in probs:
            chk.issues.append(("error" if sp != "test" else "warn", f"{sp}: {p} not found"))
        chk.splits[sp] = {"images": len(imgs), "labels": 0, "empty": 0, "missing": 0, "list": imgs,
                          "entry": d[sp]}
        total += len(imgs)
    if "train" not in chk.splits:
        chk.issues.append(("error", "No train: entry in the dataset file."))
    elif not chk.n("train"):
        chk.issues.append(("error", "The train folder has no images."))
    if "val" not in chk.splits:
        chk.issues.append(("error", "No val: entry — Ultralytics needs a validation set (make a split: Convert to "
                                    "YOLO ▸ New split)."))
    elif not chk.n("val"):
        chk.issues.append(("error", "The val folder has no images."))
    # labels
    nc = len(chk.names)
    done = 0
    seen_train = set()
    for sp, info in chk.splits.items():
        inst = Counter()
        for img in info["list"]:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            done += 1
            if progress is not None and done % 500 == 0:
                progress(done, total)
            if img.suffix.lower() not in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".dng", ".mpo",
                                          ".pfm"}:
                chk.other_formats[img.suffix.lower()] += 1
            if sp == "train":
                seen_train.add(os.path.normcase(str(img)))
            lp = _label_path(img)
            try:
                text = lp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                info["missing"] += 1
                continue
            info["labels"] += 1
            lines = [ln.split() for ln in text.splitlines() if ln.strip()]
            if not lines:
                info["empty"] += 1
            for v in lines:
                try:
                    c = int(float(v[0]))
                    vals = [float(x) for x in v[1:]]
                except (ValueError, IndexError):
                    chk.bad_lines += 1
                    continue
                if len(vals) == 4:
                    chk.fmt["box"] += 1
                    wh = (vals[2], vals[3])
                elif len(vals) >= 6 and len(vals) % 2 == 0:
                    chk.fmt["polygon"] += 1
                    xs, ys = vals[0::2], vals[1::2]
                    wh = (max(xs) - min(xs), max(ys) - min(ys))
                else:
                    chk.bad_lines += 1
                    continue
                if not 0 <= c < nc:
                    chk.bad_ids[c] += 1
                    continue
                inst[c] += 1
                if sp == "train" and len(chk.objects) < 200000:
                    chk.objects.append((img, wh))
        chk.instances[sp] = inst
    # sizes of a sample of train images (for the image size advice)
    if chk.objects:
        import random
        rng = random.Random(0)
        imgs = sorted({o[0] for o in chk.objects}, key=str)
        pick = imgs if len(imgs) <= sample_sizes else rng.sample(imgs, sample_sizes)
        from .io import image_size
        sz = {}
        for p in pick:
            try:
                s = image_size(p)
            except Exception as _exc:
                _ignored(_exc)
                s = None
            if s:
                sz[p] = s
        chk.sizes = list(sz.values())
        chk.objects = [(wh, sz[p]) for p, wh in chk.objects if p in sz]
    _issues(chk, seen_train)
    for info in chk.splits.values():
        info.pop("list", None)
    return chk


def _issues(chk: DataCheck, seen_train: set):
    d = chk.data
    tr, va = d.get("train"), d.get("val")
    if tr is not None and tr == va:
        chk.issues.append(("warn", "val is the same as train: the scores will look better than the model is. Make a "
                                   "split (Review ▸ Convert & Split ▸ Split ▸ New balanced split)."))
    for sp, info in chk.splits.items():
        n = info["images"]
        if n and info["missing"] / n > 0.5:
            chk.issues.append(("warn", f"{sp}: {info['missing']:,} of {n:,} images have no label file — Ultralytics "
                                       "treats them as background. Is the labels folder next to images?"))
    if chk.bad_ids:
        ids = ", ".join(str(k) for k in sorted(chk.bad_ids)[:6])
        chk.issues.append(("warn", f"{sum(chk.bad_ids.values()):,} objects use class ids beyond the {len(chk.names)} "
                                    f"names ({ids}) — Ultralytics drops those images. Add the names, or fix the labels."))
    if chk.bad_lines:
        chk.issues.append(("warn", f"{chk.bad_lines:,} label lines are not valid YOLO (wrong number of values)."))
    if chk.fmt["box"] and chk.fmt["polygon"]:
        chk.issues.append(("info", f"Mixed labels: {chk.fmt['box']:,} boxes and {chk.fmt['polygon']:,} polygons."))
    if chk.other_formats:
        f = ", ".join(f"{k} ({v})" for k, v in chk.other_formats.most_common(4))
        chk.issues.append(("warn", f"Images in formats older Ultralytics versions skip: {f}. Convert & Split can "
                                   "re-encode them as PNG."))
    tr_inst = chk.instances.get("train", Counter())
    va_inst = chk.instances.get("val", Counter())
    none = [chk.names[i] for i in range(len(chk.names)) if not tr_inst.get(i)]
    if none:
        chk.issues.append(("warn", f"{len(none)} class(es) have no objects in train: {', '.join(none[:5])}"
                                   + (" …" if len(none) > 5 else "") + " — the model can't learn them."))
    noval = [chk.names[i] for i in range(len(chk.names)) if tr_inst.get(i) and not va_inst.get(i)]
    if noval and "val" in chk.splits and chk.data.get("val") != chk.data.get("train"):
        chk.issues.append(("info", f"{len(noval)} class(es) have no objects in val ({', '.join(noval[:5])}"
                                   + (" …" if len(noval) > 5 else "") + "): no score for them."))
    nv = chk.n("val")
    if 0 < nv < 30:
        chk.issues.append(("info", f"Only {nv} val images: scores will jump around between epochs."))
    if tr_inst:
        rare = [chk.names[i] for i, n in tr_inst.items() if n < 10 and i < len(chk.names)]
        if rare:
            chk.issues.append(("info", f"Fewer than 10 objects in train for: {', '.join(rare[:6])}"
                                       + (" …" if len(rare) > 6 else "") + "."))


def suggest_imgsz(chk: DataCheck, min_side: float = 10.0, share: float = 0.95) -> tuple[int, str]:
    """The smallest common imgsz at which `share` of the objects keep a shorter side >= min_side px, not beyond
    the images' own size. -> (imgsz, why)."""
    if chk.cls_folder:
        return 224, "224 (what Ultralytics classifiers are trained at)"
    if not chk.objects:
        return 640, "640 (the Ultralytics default; no objects measured)"
    longs = sorted(max(w, h) for w, h in chk.sizes) or [640]
    native = longs[len(longs) // 2]
    # shorter side at imgsz = min(w*W, h*H) * imgsz / max(W, H)
    ratios = sorted(min(w * W, h * H) / max(W, H) for (w, h), (W, H) in chk.objects if W and H)
    q = ratios[int((1 - share) * (len(ratios) - 1))] if ratios else 0.0
    cands = [320, 416, 480, 512, 640, 768, 896, 1024, 1280, 1536, 1920]
    need = next((c for c in cands if q * c >= min_side), cands[-1])
    cap = max(320, int(math.ceil(native / 32) * 32))
    pick = max(320, min(need, cap))
    pick = max(pick, 640) if need <= 640 and cap >= 640 else pick
    why = (f"{pick}: {int(share * 100)} % of the objects stay ≥ {min_side:g} px across" if need <= cap else
           f"{pick}: the images' own size (median longer side {native} px); small objects would need {need} — "
           "consider tiling (SAHI) at prediction time")
    return pick, why


# ====================================================================================================== runs
def new_run_name(model: str) -> str:
    m = str(model or "model")
    if m.startswith(("arch:", "timm:", "tv:")):             # an architecture: its short name (no ":" in folders)
        m = m.split(":", 1)[1].split(".")[0]
    stem = re.sub(r'[\\/:*?"<>|\s]+', "_", Path(m).stem).strip("._") or "model"
    return f"{stem}_{time.strftime('%Y%m%d_%H%M')}"


def unique_dir(project: Path, name: str) -> Path:
    p, k = project / name, 2
    while p.exists():
        p, k = project / f"{name}_{k}", k + 1
    return p


def resolved_yaml(chk: DataCheck, dest: Path):
    """A copy of the dataset file with an absolute, existing `path` (Ultralytics resolves a relative one against
    the working directory / its datasets folder, not the file's folder)."""
    d = dict(chk.data)
    d["path"] = str(chk.root.resolve()) if chk.root else str(chk.yaml.parent.resolve())
    if chk.names:
        d["names"] = {i: n for i, n in enumerate(chk.names)}
        d["nc"] = len(chk.names)
    for k in ("download",):
        d.pop(k, None)
    lines = [f"# Copy of {chk.yaml} for this run (absolute path) — written by AIO QC"]
    lines.append(f"path: {json.dumps(d.pop('path'))}")
    for sp in ("train", "val", "test"):
        if sp in d:
            lines.append(f"{sp}: {json.dumps(d.pop(sp), ensure_ascii=False)}")
    names = d.pop("names", {})
    d.pop("nc", None)
    lines += [f"nc: {len(names)}", "names:"] + [f"  {i}: {json.dumps(str(n), ensure_ascii=False)}"
                                                for i, n in names.items()]
    for k, v in d.items():                      # anything else (kpt_shape, flip_idx, …) as JSON (valid YAML)
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_extra(text: str) -> dict:
    """'mosaic=0.5 fliplr=0 degrees=10' (spaces, commas or lines) -> {k: value} (YAML scalars)."""
    out = {}
    toks, cur, depth = [], "", 0
    for ch in text or "":                              # split on spaces / commas / semicolons outside [ ]
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        if depth == 0 and (ch.isspace() or ch in ",;"):
            toks.append(cur)
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    for tok in toks:
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"“{tok}” is not key=value")
        k, v = tok.split("=", 1)
        k = k.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            raise ValueError(f"“{k}” is not a setting name")
        out[k] = _scalar(v)
    return out


def prepare_run(python: str, chk: DataCheck, s: dict, run: Path | None = None) -> Path:
    """s: model, epochs, imgsz, batch, device, workers, patience, project, name, cache, seed, optimizer, lr0, cos_lr,
    close_mosaic, time_h, pretrained, amp, fraction, extra {..}. -> the run folder (spec + data written)."""
    project = Path(s["project"]).expanduser().resolve()
    run = run or unique_dir(project, s.get("name") or new_run_name(s["model"]))
    meta = run / "aioqc"
    meta.mkdir(parents=True, exist_ok=True)
    data = meta / "data.yaml"
    if chk.cls_folder:
        data = Path(chk.root).resolve()               # classifiers train from the folder itself
    else:
        resolved_yaml(chk, data)
    args = {"data": str(data), "epochs": int(s["epochs"]), "imgsz": int(s["imgsz"]), "batch": s["batch"],
            "device": s["device"], "workers": int(s["workers"]), "patience": int(s["patience"]),
            "project": str(project), "name": run.name, "exist_ok": True, "pretrained": bool(s.get("pretrained", True)),
            "seed": int(s.get("seed", 0)), "plots": True, "verbose": True}
    for k in ("cache", "optimizer", "lr0", "cos_lr", "close_mosaic", "amp", "fraction", "single_cls", "rect",
              "freeze"):
        if s.get(k) is not None:
            args[k] = s[k]
    if s.get("time_h"):
        args["time"] = float(s["time_h"])
    args.update(s.get("hyp") or {})                  # augmentation & schedule: only what differs from the defaults
    args.update(s.get("extra") or {})                # typed extra settings win
    engine = "torch" if is_torch_model(s["model"]) else "ultralytics"
    model = s["model"]
    if engine == "torch":                            # any PyTorch classifier: AIO QC's own trainer (cls_engine)
        model, why = resolve_arch(model, s.get("env"))
        if not model:
            raise ValueError(why)
        for k in ("cache", "close_mosaic", "rect", "single_cls", "freeze", "plots", "verbose"):
            args.pop(k, None)
        if chk.flat:
            args.setdefault("val_share", FLAT_VAL)
    elif chk.cls_folder and chk.flat:
        raise ValueError("YOLO classifiers need train/ and val/ folders — Convert & Split ▸ Class folders makes them "
                         "(or choose another architecture: it holds out validation images itself)")
    spec = {"mode": "train", "model": model, "args": args, "cwd": str(weights_dir()), "python": python,
            "engine": engine,
            "augment": s.get("augment") or None,
            "dataset_yaml": str(chk.yaml), "task": s.get("task") or chk.task, "names": chk.names,
            "created": time.time(), "settings": dict({k: v for k, v in s.items() if k not in ("extra", "augment", "env")},
                             **({"augment": ((s["augment"].get("recipe") or {}).get("name") or "recipe")}
                                if s.get("augment") else {})),
            "extra": s.get("extra") or {}}
    spec["provenance"] = _provenance(run, chk, model, args, spec["task"], "Train Model")
    (meta / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_status(meta, {"state": "queued", "created": time.time(), "model": model, "epochs": args["epochs"],
                         "save_dir": str(run), "engine": engine})
    return run


def _provenance(run: Path, chk, model, args: dict, task: str, tool: str) -> dict | None:
    """What this model will be trained from — the run folder's record now; the runner adds the weights' SHA-256
    (and puts the record into best.pt / best.onnx) when training ends."""
    from . import provenance as PV
    if not PV.enabled():
        return None
    try:
        ds = Path(chk.root) if chk.root else Path(chk.yaml).parent
        rec = PV.model_record(dataset=ds, classes=list(chk.names or []), settings=args,
                              parent=model if Path(str(model)).is_file() else None, task=task,
                              o=PV.make_origin("rule", tool))
        rec["base_model"] = PV.model_label(model)
        PV.record(run, "train", tool, inputs=[ds], input_fingerprints=False, output_fingerprint=False,
                  o=PV.make_origin("rule", tool), model=rec)
        return rec
    except Exception as exc:                          # noqa: BLE001 — a record never stops a training run
        from .diag import ignored
        ignored(exc)
        return None


def prepare_extend_run(python: str, s: dict, P, progress=None, cancel=None) -> tuple[Path, dict]:
    """Extend Model (core.extend): the training copy (old + new images, hard links) inside the run folder, the plan
    file (extend.json: lineage, resume, the command line), then the run's spec with spec["extend"].
    -> (run folder, build stats)."""
    from . import extend as X
    project = Path(s["project"]).expanduser().resolve()
    run = unique_dir(project, s.get("name") or new_run_name(s["model"]))
    meta = run / "aioqc"
    meta.mkdir(parents=True, exist_ok=True)
    tree = run / "data"
    cls = X.is_cls_folder(P.new_yaml)
    st = (X.build_cls_tree if cls else X.build_tree)(P, tree, progress=progress, cancel=cancel)
    if st.get("cancelled"):
        shutil.rmtree(run, ignore_errors=True)
        return run, st
    (meta / "tree.json").write_text(json.dumps({"new_train_labels": st.pop("new_train_labels")}), encoding="utf-8")
    P.save(meta / "extend.json")
    chk = check_dataset(tree if cls else tree / "data.yaml")
    s = dict(s, model=P.base, task=s.get("task") or chk.task)
    prepare_run(python, chk, s, run=run)
    spec = json.loads((meta / "spec.json").read_text(encoding="utf-8"))
    spec["args"]["data"] = str(tree if cls else tree / "data.yaml")
    spec["meta"] = str(meta)
    names, old_of = P.names, P.old_of
    if cls:                       # a classifier's ids are its class folders in name order: rows copied by name
        names = sorted(P.names)
        old_of = {names.index(n): P.base_names.index(n) for n in P.base_names if n in names}
    if spec.get("engine") == "torch":           # a PyTorch classifier: its trainer matches the classes by name; two
        if P.two_stage:                        # stages = the head alone first
            spec["args"]["freeze_epochs"] = int(P.stage1_epochs or 5)
        spec["args"].pop("val_share", None)
    spec["extend"] = {"base": P.base, "base_names": P.base_names, "names": names, "cls": cls,
                      "old_of": {str(k): v for k, v in old_of.items()}, "goal": P.goal, "tree": str(tree),
                      "old_yaml": P.old_yaml, "new_yaml": P.new_yaml, "prelabel": P.prelabel and not cls,
                      "prelabel_conf": P.prelabel_conf, "review": P.review, "two_stage": P.two_stage,
                      "stage1_epochs": P.stage1_epochs, "freeze": P.freeze, "stage2_lr": P.stage2_lr,
                      "forget_limit": P.forget_limit}
    (meta / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    st["objects"] = dict(st["objects"])
    st["images"] = dict(st["images"])
    st["how"] = dict(st["how"])
    (meta / "tree_stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    (meta / "README.txt").write_text(
        "Extend Model run, planned by AIO QC\n\n"
        "extend.json   the plan: the base model, the classes (old ids locked, new ones appended), the data, the stages\n"
        "spec.json     what the runner does (Train Model starts it; Continue after a review)\n"
        "tree.json     the new training images whose labels get the pre-labels\n\n"
        "Without the app (e.g. on a training server with the same folders):\n"
        "    <python with torch + ultralytics> -u train_runner.py <this folder>/spec.json\n"
        "train_runner.py is in the AIO QC package (AIO_QC/core/train_runner.py) and needs nothing else.\n"
        "Pre-labels waiting for a review: set \"review\": false under \"extend\" in spec.json, then run it again.\n",
        encoding="utf-8")
    try:                                              # lineage: which model each extended one came from
        with open(project / "lineage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(X.history_line(P, run, None), ensure_ascii=False) + "\n")
    except OSError:
        pass
    return run, st


def prepare_continue(run: Path) -> Path:
    """After the pre-labels were reviewed: the same run goes on (pre-labels kept, review not asked again)."""
    meta = run / "aioqc"
    spec = json.loads((meta / "spec.json").read_text(encoding="utf-8"))
    spec.setdefault("extend", {})["review"] = False
    (meta / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_status(meta, {"state": "queued", "created": time.time(), "model": spec.get("model"),
                         "save_dir": str(run)})
    return run


def extend_result(run) -> dict | None:
    """The before / after comparison of an Extend Model run (compare.json, core.extend.compare)."""
    from . import extend as X
    meta = Path(run) / "aioqc"
    try:
        raw = json.loads((meta / "compare.json").read_text(encoding="utf-8"))
        P = X.ExtendPlan.load(meta / "extend.json")
    except (OSError, ValueError, TypeError):
        return None
    return X.compare(raw.get("before") or {}, raw.get("after") or {}, P.base_names, P.forget_limit)


def prepare_resume(run: Path, python: str) -> Path:
    meta = run / "aioqc"
    spec = json.loads((meta / "spec.json").read_text(encoding="utf-8")) if (meta / "spec.json").exists() else {}
    spec.update({"mode": "resume", "resume_from": str(run / "weights" / "last.pt"), "python": python,
                 "cwd": str(weights_dir())})
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    return run


def prepare_val(model: str, data_yaml: str, out_dir: Path, python: str, split="val", imgsz=640, device="",
                batch=16, extra: dict | None = None) -> Path:
    """A check of `model` on one split. A PyTorch classifier (is_torch_model) is checked by its own trainer — with
    the same held-out images of a flat folder (extra: {"val_share": …} of its run)."""
    out_dir = unique_dir(out_dir.parent, out_dir.name)
    meta = out_dir / "aioqc"
    meta.mkdir(parents=True, exist_ok=True)
    args = {"data": data_yaml, "split": split, "imgsz": int(imgsz), "batch": batch, "plots": True,
            "project": str(out_dir.parent), "name": out_dir.name, "exist_ok": True}
    if device != "":
        args["device"] = device
    args.update(extra or {})
    spec = {"mode": "val", "model": model, "args": args, "cwd": str(weights_dir()), "python": python}
    if is_torch_model(model):
        spec["engine"] = "torch"
    (meta / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_status(meta, {"state": "queued", "created": time.time(), "model": model, "mode": "val"})
    return out_dir


def _write_status(meta: Path, d: dict):
    tmp = meta / "status.json.tmp"
    tmp.write_text(json.dumps(d), encoding="utf-8")
    os.replace(tmp, meta / "status.json")


# ------------------------------------------------------------------ launching
_live: list = []


def runner_copy() -> Path:
    """The runner, copied to a folder of its own (sys.path[0] of the run is that folder: none of AIO QC's module
    names can shadow the environment's packages)."""
    d = Path(os.environ["AIOQC_RUNNER_DIR"]) if os.environ.get("AIOQC_RUNNER_DIR") else data_home() / "runner"
    d.mkdir(parents=True, exist_ok=True)              # (AIOQC_RUNNER_DIR: test runs, so coverage can find it)
    dst = d / "train_runner.py"
    # … and the augmentation engine next to it (one standalone file): recipes run in memory while training
    for target, source in ((dst, RUNNER), (d / "aioqc_augment.py", RUNNER.with_name("augment_ops.py")),
                           (d / "aioqc_cls_engine.py", RUNNER.with_name("cls_engine.py"))):
        src = source.read_bytes()
        if not target.exists() or hashlib.sha1(target.read_bytes()).digest() != hashlib.sha1(src).digest():
            target.write_bytes(src)
    return dst


def _env() -> dict:
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
              "QML2_IMPORT_PATH", "QT_API", "QT_QPA_PLATFORM"):
        env.pop(k, None)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "TQDM_MININTERVAL": "2"})
    return env


def launch(run: Path, python: str) -> dict:
    """Start the run detached. -> {"pid", "survives": the run keeps going when AIO QC closes}."""
    meta = run / "aioqc"
    cmd = [python, "-u", str(runner_copy()), str(meta / "spec.json")]
    log = open(meta / "train.log", "ab")
    survives = True
    try:
        kw = dict(stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(weights_dir()), env=_env())
        if WIN:
            flags = 0x08000000 | 0x00000200            # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
            try:
                p = subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)   # + CREATE_BREAKAWAY_FROM_JOB
            except OSError:
                p = subprocess.Popen(cmd, creationflags=flags, **kw)
                survives = False                      # inside a job that forbids leaving: ends with the app
        else:
            p = subprocess.Popen(cmd, start_new_session=True, **kw)
    finally:
        log.close()
    _live.append(p)
    st = status(run)
    state = st.get("state", "queued")
    if state in ENDED or state == "review":            # resumed / continued: queued until the runner reports
        state = "queued"
        for k in ("ended", "message", "error", "error_tail", "request"):
            st.pop(k, None)
    st.update({"pid": p.pid, "state": state, "survives": survives, "launched": time.time()})
    _write_status(meta, st)
    return {"pid": p.pid, "survives": survives}


def reap():
    """Collect finished children (no zombies)."""
    for p in list(_live):
        if p.poll() is not None:
            _live.remove(p)


def probe(python: str, timeout: float = 180) -> dict:
    """Versions and devices of an environment (imports torch: can take a while)."""
    from . import procs
    r = procs.run([python, "-u", str(runner_copy()), "--probe"], timeout=timeout, text=True)
    for ln in reversed((r.stdout or "").splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            return json.loads(ln)
    raise RuntimeError((r.stderr or r.stdout or "no output").strip()[-800:])


# ------------------------------------------------------------------ reading a run
def status(run) -> dict:
    p = Path(run) / "aioqc" / "status.json"
    for _ in range(3):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            time.sleep(0.03)
    return {}


def state_of(st: dict, now: float | None = None) -> str:
    """The run's state, with a stale heartbeat seen for what it is ("interrupted": the process is gone)."""
    s = st.get("state", "")
    if s in RUNNING:
        age = (now or time.time()) - float(st.get("updated") or st.get("launched") or 0)
        if age > 90 and not pid_alive(st.get("pid")):
            return "interrupted"
        if age > 90:
            return "not responding"
    if s == "queued" and (now or time.time()) - float(st.get("launched") or st.get("created") or 0) > 120:
        return "interrupted" if not pid_alive(st.get("pid")) else s
    return s


def pid_alive(pid) -> bool:
    if not pid:
        return False
    pid = int(pid)
    if WIN:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                               # a zombie child of ours counts as gone
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def epochs(run) -> list[dict]:
    """Per-epoch rows: AIO QC's own log, else Ultralytics' results.csv (runs started elsewhere)."""
    p = Path(run) / "aioqc" / "epochs.jsonl"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
        if rows:
            return rows
    return results_csv(run)


def results_csv(run) -> list[dict]:
    p = Path(run) / "results.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in r.items() if k}
            try:
                ep = int(float(r.get("epoch", 0)))
            except ValueError:
                continue
            row = {"epoch": ep, "loss": {}, "metrics": {}, "lr": {}}
            for k, v in r.items():
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if k.startswith("train/"):
                    row["loss"][k] = fv
                elif k.startswith(("metrics/", "val/")):
                    row["metrics"][k] = fv
                elif k.startswith("lr/"):
                    row["lr"][k] = fv
                elif k == "time":
                    row["t_total"] = fv
            rows.append(row)
    return rows


def log_tail(run, max_bytes: int = 60000) -> str:
    p = Path(run) / "aioqc" / "train.log"
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return clean_log(data)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def clean_log(text: str) -> str:
    """Colour codes out; progress bars redrawn with \\r keep only their last state."""
    out = []
    for ln in _ANSI.sub("", text).replace("\r\n", "\n").split("\n"):
        if "\r" in ln:
            ln = [x for x in ln.split("\r") if x.strip()][-1:] or [""]
            ln = ln[0]
        out.append(ln)
    return "\n".join(out)


def control(run, what: str):
    """what: stop (finish now, keep results) | pause (after the next saved epoch, resumable) | kill (at once)."""
    (Path(run) / "aioqc" / what).write_text(str(time.time()), encoding="utf-8")


def hard_kill(pid) -> bool:
    if not pid:
        return False
    try:
        if WIN:
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True,
                           creationflags=0x08000000, timeout=20)
        else:
            import signal
            try:
                os.killpg(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                os.kill(int(pid), signal.SIGKILL)
        return True
    except Exception as _exc:
        _ignored(_exc)
        return False


ENDED = {"finished", "stopped", "paused", "failed", "interrupted"}


def is_runner(pid) -> bool:
    """Is `pid` still a training runner (not a recycled process id)? Unknown: False — never kill a stranger."""
    if not pid_alive(pid):
        return False
    try:
        import psutil
        return any("train_runner.py" in str(a) for a in psutil.Process(int(pid)).cmdline())
    except ImportError:
        pass
    except Exception as _exc:
        _ignored(_exc)
        return False
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return b"train_runner.py" in f.read()
    except OSError:
        return False


def ensure_released(run, grace: float = 20.0) -> bool:
    """A run that has ended (failed, stopped, finished …) must not keep its process — and with it GPU memory or data
    workers. If it is still there `grace` seconds after it ended, its whole process tree is killed. -> killed."""
    st = status(run)
    if not st or state_of(st) not in ENDED | {"not responding"}:
        return False
    pid = st.get("pid")
    if not is_runner(pid):
        return False
    t_end = float(st.get("ended") or st.get("updated") or 0)
    if state_of(st) == "not responding":
        return False                                   # the user decides (End in the running-training list)
    if time.time() - t_end < grace:
        return False
    hard_kill(pid)
    return True


def end_run(run, wait: float = 6.0) -> None:
    """End a run for good, now: ask (kill file), then kill the process tree if it is still there."""
    st = status(run)
    control(run, "kill")
    pid = st.get("pid")
    t = time.time()
    while time.time() - t < wait and is_runner(pid):
        time.sleep(0.2)
    if is_runner(pid):
        hard_kill(pid)
        st = status(run)
        if st.get("state") in RUNNING | {"queued"}:
            st.update(state="stopped", message="Ended (the process was killed; the last saved epoch can be "
                                               "resumed)", resumable=True, ended=time.time())
            _write_status(Path(run) / "aioqc", st)


def list_runs(project) -> list[dict]:
    """Runs in a project folder, newest first: {dir, name, state, model, epoch, epochs, best (mAP50-95), map50,
    started, resumable, task}."""
    out = []
    p = Path(project)
    if not p.is_dir():
        return out
    for d in p.iterdir():
        if not d.is_dir():
            continue
        st = status(d)
        has = st or (d / "results.csv").exists() or (d / "weights").is_dir()
        if not has:
            continue
        spec = {}
        try:
            spec = json.loads((d / "aioqc" / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        rows = epochs(d)
        best, b50 = best_scores(rows)
        state = state_of(st) if st else ("finished" if (d / "weights" / "best.pt").exists() else "unknown")
        last = d / "weights" / "last.pt"
        out.append({"dir": d, "name": d.name, "state": state, "model": spec.get("model") or st.get("model", ""),
                    "mode": spec.get("mode", "train"),
                    "epoch": max((r.get("epoch", 0) for r in rows if not r.get("final")), default=0),
                    "epochs": st.get("epochs") or (spec.get("args") or {}).get("epochs"),
                    "best": best, "map50": b50, "started": st.get("started") or st.get("created") or d.stat().st_mtime,
                    "metric": score_word(is_cls_rows(rows) or spec.get("task") == "cls"),
                    "resumable": last.exists() and state in ("paused", "stopped", "interrupted", "failed")
                    and st.get("resumable", True) is not False, "task": spec.get("task", ""),
                    "engine": spec.get("engine") or "ultralytics",
                    "best_pt": str(d / "weights" / "best.pt") if (d / "weights" / "best.pt").exists() else ""})
    out.sort(key=lambda r: -float(r["started"] or 0))
    return out


MAP_KEYS = ("metrics/mAP50-95(M)", "metrics/mAP50-95(B)", "metrics/accuracy_top1")
TOP1, TOP5 = "metrics/accuracy_top1", "metrics/accuracy_top5"


def is_cls_rows(rows) -> bool:
    """A classifier's run: its epochs report top-1 / top-5 accuracy instead of mAP."""
    return any(TOP1 in (r.get("metrics") or {}) for r in rows)


def score_word(cls: bool) -> str:
    return "top-1" if cls else "mAP50-95"


def best_scores(rows) -> tuple[float | None, float | None]:
    """(best main score, its companion): mAP50-95 + mAP50 for detection / segmentation, top-1 + top-5 accuracy for
    a classifier."""
    best = b50 = None
    for r in rows:
        m = r.get("metrics") or {}
        for k in MAP_KEYS:
            if k in m:
                v = float(m[k])
                if best is None or v > best:
                    best = v
                    b50 = m.get(TOP5) if k == TOP1 else m.get(k.replace("50-95", "50"))
                break
    return best, b50


HINTS = [
    (r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory",
     "The GPU ran out of memory — lower the batch (or choose Auto), the image size, or pick a smaller model."),
    (r"DataLoader worker .* (killed|exited unexpectedly)|shared memory|Bus error|No space left on device \(shm\)",
     "A data-loading worker ran out of memory — lower Workers, and turn image caching off."),
    (r"MemoryError|Unable to allocate|cannot allocate memory",
     "The computer ran out of RAM — turn image caching off (or use disk), lower Workers or the batch."),
    (r"GatedRepoError|gated repo|is restricted\. You must|401 Client Error",
     "These pretrained weights are gated on Hugging Face: open the model's page there (huggingface.co/timm/<model>), "
     "accept its licence with your account, then sign in on this computer — run “hf auth login” in this environment, "
     "or set the HF_TOKEN environment variable before starting AIO QC (the token stays in the environment; AIO QC "
     "never stores it) — and start again."),
    (r"Unknown model|Invalid pretrained tag|No pretrained weights exist",
     "This timm does not know that model or its pretrained tag — update timm (python -m pip install -U timm in this "
     "environment; Train Model offers it for the built-in architectures) or check the name (timm.list_pretrained())."),
    (r"ConnectionError|Download failure|urlopen error|getaddrinfo|Failed to download|HTTP Error 40\d|"
     r"Max retries exceeded",
     "A download failed (the model weights or a font) — no internet? Choose a .pt file you already have "
     "(Browse…), or connect and try again."),
    (r"huggingface|hf_hub|LocalEntryNotFoundError|HFValidationError|download\.pytorch\.org|"
     r"Couldn't reach|offline mode",
     "The pretrained weights could not be downloaded (no internet, or the site is blocked) — connect and try again, "
     "or untick Pretrained weights (needs far more data), or start from a model file you have."),
    (r"No module named 'ultralytics'|No module named 'torch'",
     "This Python environment lacks PyTorch / Ultralytics — choose another one, or install them (Models & "
     "Environment ▸ Environment)."),
    (r"Torch not compiled with CUDA enabled|Invalid CUDA 'device=|CUDA is not available|no CUDA GPUs",
     "This environment's PyTorch can't use that GPU — choose CPU, or install a CUDA build of PyTorch."),
    (r"Dataset .* (images|images not found)|does not exist|missing path|not found",
     "A dataset path is wrong — run the dataset check again; it lists the paths it could not find."),
    (r"Label class \d+ exceeds|exceeds dataset class count",
     "Some labels use class ids beyond the class names — fix the labels or the names."),
    (r"No labels found|no labels",
     "No labels were found — the labels folder must sit next to images (…/images → …/labels)."),
    (r"is not a valid YOLO argument|SyntaxError|unexpected keyword|Invalid .* argument",
     "One of the settings is not valid for this Ultralytics version — check Extra settings."),
    (r"Resume checkpoint .* not found|is not resumable|nothing to resume|training to \d+ epochs is finished",
     "This run can't be resumed (it finished, or its last.pt was made by a final check) — start a new run "
     "from its best.pt instead."),
]


def missing_module(text: str) -> str | None:
    """The package a 'No module named …' error names (pip name for the usual ones)."""
    m = re.search(r"No module named '([\w.\-]+)'", text or "")
    if not m:
        return None
    mod = m.group(1).split(".")[0]
    return {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn",
            "pynvml": "nvidia-ml-py", "thop": "ultralytics-thop"}.get(mod, mod)


def explain(text: str) -> str:
    pkg = missing_module(text)
    if pkg and pkg not in ("ultralytics", "torch"):
        return (f"This Python environment lacks “{pkg}”, which this training needs — choose Install and retry (or run: "
                f"python -m pip install {pkg} in it), then Resume or start again.")
    for rx, hint in HINTS:
        if re.search(rx, text or "", re.I):
            return hint
    return ""


def pip_install(python: str, packages: list[str], timeout: float = 1800) -> tuple[bool, str]:
    """python -m pip install <packages> in that environment -> (ok, output tail)."""
    from . import procs
    r = procs.run([python, "-m", "pip", "install", *packages], timeout=timeout, text=True)
    out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    return r.returncode == 0, out[-3000:]


def env_summary(info: dict) -> str:
    gpus = [d for d in info.get("devices", []) if d["id"] not in ("cpu",)]
    return (f"Python {info.get('python')} · torch {info.get('torch') or '—'} · ultralytics "
            f"{info.get('ultralytics') or '—'} · " + (", ".join(d["name"] for d in gpus) if gpus else "CPU only"))


def free_disk_gb(path) -> float:
    try:
        p = Path(path)
        while not p.exists():
            p = p.parent
        return shutil.disk_usage(p).free / 2**30
    except OSError:
        return float("inf")
