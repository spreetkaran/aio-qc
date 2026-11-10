# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Shared constants and small helpers (Qt-free) - one definition used everywhere.
"""
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".wmv", ".mpeg", ".mpg", ".m4v", ".ts", ".mts")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2", ".avif")
PT_EXTS = (".pt", ".engine", ".torchscript")              # run in a Python environment (ultralytics)
WEIGHT_EXTS = (".onnx",) + PT_EXTS


# One colour per model everywhere (overlay, Model Lab, charts, reports, annotated videos) - a validated
# categorical palette (colour-blind safe adjacent pairs), with separate steps for light and dark backgrounds.
MODEL_PALETTE = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
                 "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}


def model_hex(slot, mode="dark"):
    return MODEL_PALETTE[mode][int(slot) % 8]


def file_filter(label, exts, all_files=True):
    """'Videos (*.mp4 *.avi ...);;All files (*)' for Qt file dialogs."""
    f = f"{label} ({' '.join('*' + e for e in exts)})"
    return f + ";;All files (*)" if all_files else f


VIDEO_FILTER = file_filter("Videos", VIDEO_EXTS)
IMAGE_FILTER = file_filter("Images", IMG_EXTS)
WEIGHTS_FILTER = file_filter("YOLO weights", WEIGHT_EXTS)


def is_video(path):
    return str(path).lower().endswith(VIDEO_EXTS)


def is_image(path):
    return str(path).lower().endswith(IMG_EXTS)


def parse_range(text):
    """'1:00-4:30', '60-270', '90' (start only), '' / 'whole video' -> (start, end) seconds (None = open)."""
    from AIO_QC.tools.frame_extractor.core.sources import parse_time
    text = (text or "").strip().lower()
    if not text or text.startswith("whole") or text in ("–", "-"):
        return None, None
    a, sep, b = text.partition("-")
    return (parse_time(a) if a.strip() else None), (parse_time(b) if sep and b.strip() else None)


def short_time(sec):
    """Compact display time: 1:02.50 or 1:01:02.50."""
    sec = max(0.0, float(sec or 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}" if h else f"{int(m)}:{s:05.2f}"
