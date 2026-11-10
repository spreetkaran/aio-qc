# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Text on plotted images (box labels, HUD) in Inter - Qt-free, used by core.lab.annotate for the app's galleries,
saved plots, reports and annotated videos.

Glyphs are rasterised with Pillow/FreeType from the bundled Inter (FrameExtractor/fonts) and alpha-blended onto
the BGR image. Rendered strings are cached (labels repeat frame after frame), so drawing costs a few
microseconds per label after the first time. Without Pillow or the font files the OpenCV Hershey font is used -
nothing breaks.

  text_size(text, px, weight)            -> (width, height) of the label box content
  put_text(img, text, x, y, px, color, weight)   draws with (x, y) = top-left of the text box
"""
import os
import threading
from collections import OrderedDict

import cv2
import numpy as np

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__import__("AIO_QC").__file__)), "fonts")   # AIO QC's font folder (Qt-free)
_fonts = {}
_cache = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 4096
_ok = None


def available():
    """True when Inter can be used (Pillow + the bundled font)."""
    global _ok
    if _ok is None:
        try:
            from PIL import ImageFont  # noqa: F401
            _ok = os.path.isfile(os.path.join(FONT_DIR, "Inter-SemiBold.otf"))
        except Exception:
            _ok = False
    return _ok


def _font(px, weight):
    key = (px, weight)
    f = _fonts.get(key)
    if f is None:
        from PIL import ImageFont
        path = os.path.join(FONT_DIR, f"Inter-{weight}.otf")
        if not os.path.isfile(path):
            path = os.path.join(FONT_DIR, "Inter-Regular.otf")
        f = _fonts[key] = ImageFont.truetype(path, px)
    return f


def _px(px):
    return int(max(7, min(160, round(px))))


def _mask(text, px, weight):
    """(alpha uint8 [h, w], ascent) of a string - line box is the font's ascent + descent, so every label of one
    size has the same height whatever its letters."""
    key = (text, px, weight)
    with _lock:
        m = _cache.get(key)
        if m is not None:
            _cache.move_to_end(key)
            return m
    f = _font(px, weight)
    asc, desc = f.getmetrics()
    x0, _, x1, _ = f.getbbox(text)
    w = max(1, int(np.ceil(x1 - min(0, x0))) + 1)
    h = asc + desc
    from PIL import Image, ImageDraw
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-min(0, x0), 0), text, font=f, fill=255)
    m = (np.asarray(im, np.uint8), asc)
    with _lock:
        _cache[key] = m
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return m


def _hershey_scale(px):
    return px / 22.0


def text_size(text, px, weight="SemiBold"):
    """(w, h) of the text's box at px pixels (cap-to-descender line height)."""
    px = _px(px)
    if available():
        a, _ = _mask(text, px, weight)
        return a.shape[1], a.shape[0]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, _hershey_scale(px), 1)
    return tw, th + base


def put_text(img, text, x, y, px, color=(255, 255, 255), weight="SemiBold"):
    """Draw text with its box's top-left at (x, y); clipped to the image. color is BGR."""
    px = _px(px)
    x, y = int(round(x)), int(round(y))
    if not available():
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, _hershey_scale(px), 1)
        cv2.putText(img, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, _hershey_scale(px), color,
                    max(1, int(px / 14)), cv2.LINE_AA)
        return
    a, _ = _mask(text, px, weight)
    H, W = img.shape[:2]
    h, w = a.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    alpha = a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] * (1 / 255.0)
    roi = img[y0:y1, x0:x1]
    col = np.asarray(color, np.float32).reshape(1, 1, -1)[..., :roi.shape[2] if roi.ndim == 3 else 1]
    if roi.ndim == 2:
        alpha = alpha[..., 0]
        col = float(np.mean(color))
    roi[...] = (roi * (1 - alpha) + col * alpha + 0.5).astype(np.uint8)
