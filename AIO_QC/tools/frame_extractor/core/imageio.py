# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image input / output: formats, quality, unicode-safe reading and writing (Qt-free).

One quality value (0-100, default 95) is mapped to each encoder:
  JPEG       quality; >= 90 also keeps full colour resolution (4:4:4 chroma - sharp edges on plates/text),
             optimised Huffman tables (smaller, same pixels)
  WebP       quality; 100 = lossless
  AVIF       quality; 100 = near-lossless
  JPEG 2000  compression rate (100 = best)
  PNG / TIFF / BMP   always lossless (quality ignored; PNG uses fast compression, TIFF uses LZW)
"""
import os
from functools import lru_cache

import cv2
import numpy as np

# ext: (label, lossy, note)
FORMATS = {
    ".jpg": ("JPEG", True, "Universal; smallest decode cost. Q ≥ 90 keeps full colour detail (4:4:4)."),
    ".png": ("PNG", False, "Lossless. Larger files; exact pixels."),
    ".webp": ("WebP", True, "~30 % smaller than JPEG at equal quality. Q 100 = lossless."),
    ".avif": ("AVIF", True, "Smallest files; slow to encode; check your tools can read it."),
    ".jp2": ("JPEG 2000", True, "Good quality at high compression; limited tool support."),
    ".tif": ("TIFF", False, "Lossless (LZW). Common in imaging pipelines."),
    ".bmp": ("BMP", False, "Uncompressed; very large."),
}
DEFAULT_EXT, DEFAULT_QUALITY = ".jpg", 95


@lru_cache(maxsize=None)
def available_formats():
    """Extensions this OpenCV build can encode (tested once)."""
    probe = np.full((64, 64, 3), 128, np.uint8)   # JPEG 2000 rejects tiny images
    out = []
    for ext in FORMATS:
        try:
            ok, _ = cv2.imencode(ext, probe, encode_params(ext, 90))
            if ok:
                out.append(ext)
        except cv2.error:
            pass
    return out


def normalize_ext(ext):
    ext = (ext or DEFAULT_EXT).lower()
    ext = ext if ext.startswith(".") else "." + ext
    return {".jpeg": ".jpg", ".tiff": ".tif", ".j2k": ".jp2"}.get(ext, ext)


def is_lossy(ext):
    return FORMATS.get(normalize_ext(ext), ("", True, ""))[1]


def encode_params(ext, quality=DEFAULT_QUALITY):
    ext, q = normalize_ext(ext), int(max(0, min(100, quality)))
    p = []
    if ext == ".jpg":
        p = [cv2.IMWRITE_JPEG_QUALITY, max(1, q), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        if q >= 90 and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
            p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    elif ext == ".webp":
        p = [cv2.IMWRITE_WEBP_QUALITY, 101 if q >= 100 else max(1, q)]
    elif ext == ".avif" and hasattr(cv2, "IMWRITE_AVIF_QUALITY"):
        p = [cv2.IMWRITE_AVIF_QUALITY, q]
        if hasattr(cv2, "IMWRITE_AVIF_SPEED"):
            p += [cv2.IMWRITE_AVIF_SPEED, 8]          # 0 = slowest/best ... 10 = fastest
    elif ext == ".jp2":
        p = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, max(1, q * 10)]
    elif ext == ".png":
        p = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext == ".tif" and hasattr(cv2, "IMWRITE_TIFF_COMPRESSION"):
        p = [cv2.IMWRITE_TIFF_COMPRESSION, getattr(cv2, "IMWRITE_TIFF_COMPRESSION_LZW", 5)]
    return p


def quality_note(ext, quality):
    ext, q = normalize_ext(ext), int(quality)
    if not is_lossy(ext):
        return "lossless — quality not used"
    if ext == ".webp" and q >= 100:
        return "lossless"
    if ext == ".jpg":
        return ("maximum" if q >= 98 else "visually lossless" if q >= 90 else "high" if q >= 80 else
                "medium" if q >= 60 else "low") + (" · 4:4:4 colour" if q >= 90 else " · 4:2:0 colour")
    return "maximum" if q >= 95 else "high" if q >= 80 else "medium" if q >= 60 else "low"


def encode(img, ext=DEFAULT_EXT, quality=DEFAULT_QUALITY):
    ext = normalize_ext(ext)
    ok, buf = cv2.imencode(ext, img, encode_params(ext, quality))
    if not ok:
        raise IOError(f"Could not encode image as {ext}")
    return buf


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-ASCII paths on Windows (and never raises on a bad file)."""
    try:
        img = cv2.imread(path, flags)
        if img is None and os.path.isfile(path):
            img = cv2.imdecode(np.fromfile(path, np.uint8), flags)
        return img
    except Exception:
        return None


def imwrite(path, img, ext=None, quality=DEFAULT_QUALITY):
    """cv2.imwrite replacement: any supported format, quality, non-ASCII Windows paths."""
    ext = normalize_ext(ext or os.path.splitext(path)[1])
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    encode(img, ext, quality).tofile(path)


def encoded_size(img, ext, quality):
    """Bytes this frame would take (for the size estimate in the settings)."""
    return int(encode(img, ext, quality).size)
