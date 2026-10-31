# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Image Quality: integrity first (errors), then low-priority information.

    integrity(path)                 -> [(code, severity, message)]   structural, cheap (no full decode)
    exif_and_size(path, ann_w, ann_h) -> [(code, severity, message)]  EXIF orientation vs annotation, size mismatch
    metrics(path)                   -> {blur, mean, std, dark, bright}  on a ~512 px reduced decode
    judge_metrics(m, thr)           -> [(code, "info", message)]

JPEG: the marker structure is walked exactly like libjpeg does, so the same problems it only prints as warnings are
found: "Corrupt JPEG data: N extraneous bytes before marker 0xXX", "Premature end of JPEG file" (truncated), no
end-of-image marker, data after the end. PNG: chunks + CRCs + IEND. Other formats: Pillow verify().
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

ERROR, WARN, INFO = "error", "warn", "info"
DEFAULT_THRESHOLDS = {"blur": 40.0, "dark_mean": 45.0, "bright_mean": 215.0, "low_std": 18.0,
                      "dark_frac": 0.65, "bright_frac": 0.45}
TITLES = {
    "unreadable": "unreadable", "empty_file": "empty file", "truncated": "truncated", "corrupt": "corrupt data",
    "trailing": "data after the end", "bad_crc": "corrupt (CRC)", "decode": "cannot be decoded",
    "exif_rotated": "EXIF rotation mismatch", "exif_present": "EXIF rotation", "size_mismatch": "size mismatch",
    "blur": "blurry", "dark": "under-exposed", "bright": "over-exposed", "low_contrast": "low contrast",
}
SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2}


# =============================================================================
# Integrity (structural)
# =============================================================================
def _jpeg(data: bytes) -> list[tuple]:
    out = []
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return [("corrupt", ERROR, "not a JPEG (no start-of-image marker)")]
    i = 2
    eoi = None
    while i < n:
        if data[i] != 0xFF:                               # garbage where a marker must start
            j = data.find(b"\xff", i)
            if j < 0:
                out.append(("truncated", ERROR, "Premature end of JPEG file (no end-of-image marker)"))
                return out
            nxt = data[j + 1] if j + 1 < n else 0
            out.append(("corrupt", ERROR, f"Corrupt JPEG data: {j - i} extraneous bytes before marker 0x{nxt:02x}"))
            i = j
            continue
        while i < n and data[i] == 0xFF:                  # fill bytes
            i += 1
        if i >= n:
            break
        m = data[i]
        i += 1
        if m == 0xD9:
            eoi = i
            break
        if m in (0x01,) or 0xD0 <= m <= 0xD7:
            continue
        if i + 2 > n:
            out.append(("truncated", ERROR, "Premature end of JPEG file (cut inside a header)"))
            return out
        length = struct.unpack(">H", data[i:i + 2])[0]
        if length < 2 or i + length > n:
            out.append(("truncated", ERROR, f"Premature end of JPEG file (segment 0x{m:02x} runs past the end)"))
            return out
        i += length
        if m == 0xDA:                                     # entropy-coded data: skip to the next real marker
            while True:
                j = data.find(b"\xff", i)
                if j < 0 or j + 1 >= n:
                    out.append(("truncated", ERROR, "Premature end of JPEG file (image data is cut off)"))
                    return out
                nb = data[j + 1]
                if nb == 0x00 or 0xD0 <= nb <= 0xD7 or nb == 0xFF:
                    i = j + 2 if nb != 0xFF else j + 1
                    continue
                i = j
                break
    if eoi is None:
        out.append(("truncated", ERROR, "Premature end of JPEG file (no end-of-image marker)"))
    elif n - eoi > 0 and data[eoi:].strip(b"\x00"):
        out.append(("trailing", INFO, f"{n - eoi:,} bytes after the end-of-image marker"))
    return out


def _png(data: bytes) -> list[tuple]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [("corrupt", ERROR, "not a PNG (bad signature)")]
    i, n = 8, len(data)
    while i + 8 <= n:
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        end = i + 12 + length
        if end > n:
            return [("truncated", ERROR, f"PNG is truncated (chunk {ctype.decode('latin-1', 'replace')} cut off)")]
        crc = struct.unpack(">I", data[i + 8 + length:end])[0]
        if zlib.crc32(data[i + 4:i + 8 + length]) & 0xFFFFFFFF != crc:
            return [("bad_crc", ERROR, f"PNG chunk {ctype.decode('latin-1', 'replace')} has a bad CRC (corrupt data)")]
        if ctype == b"IEND":
            rest = data[end:]
            return [("trailing", INFO, f"{len(rest):,} bytes after IEND")] if rest.strip(b"\x00") else []
        i = end
    return [("truncated", ERROR, "PNG is truncated (no IEND chunk)")]


def integrity(path, decode: bool = False) -> list[tuple]:
    """Structural check (fast). decode=True also decodes the whole image (catches broken compressed data)."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        return [("unreadable", ERROR, f"cannot read the file: {exc}")]
    if not data:
        return [("empty_file", ERROR, "the file is empty (0 bytes)")]
    head = data[:8]
    if head[:2] == b"\xff\xd8":
        out = _jpeg(data)
    elif head.startswith(b"\x89PNG"):
        out = _png(data)
    else:
        out = []
        try:
            from PIL import Image
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:
            out.append(("corrupt", ERROR, f"Pillow cannot verify the image: {exc}"))
    if decode and not any(s == ERROR for _c, s, _m in out):
        try:
            from io import BytesIO
            from PIL import Image
            with Image.open(BytesIO(data)) as im:
                im.load()
        except Exception as exc:
            out.append(("decode", ERROR, f"decoding failed: {exc}"))
    return out


# =============================================================================
# EXIF orientation + size against the annotation
# =============================================================================
ROTATED = {5, 6, 7, 8}
ORIENT_TEXT = {2: "mirrored", 3: "rotated 180°", 4: "flipped", 5: "transposed", 6: "rotated 90°",
               7: "transversed", 8: "rotated 270°"}


def exif_info(path) -> tuple[int, int, int] | None:
    """(raw width, raw height, EXIF orientation) without decoding the pixels."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            o = 1
            try:
                o = int(im.getexif().get(0x0112, 1) or 1)
            except Exception:
                pass
            return im.width, im.height, o
    except Exception:
        return None


def exif_and_size(path, ann_w: float | None, ann_h: float | None) -> list[tuple]:
    info = exif_info(path)
    if info is None:
        return []
    w, h, o = info
    shown = (h, w) if o in ROTATED else (w, h)               # what EXIF-aware tools display
    out = []
    if ann_w and ann_h:
        aw, ah = int(round(ann_w)), int(round(ann_h))
        if o in ROTATED and (aw, ah) == (w, h) and w != h:
            out.append(("exif_rotated", ERROR, f"EXIF says {ORIENT_TEXT[o]}, but the annotation ({aw}×{ah}) was made on "
                                               f"the un-rotated pixels — tools that apply EXIF (labelme, Ultralytics, "
                                               f"most viewers) show the boxes rotated"))
        elif (aw, ah) != shown and (aw, ah) != (w, h):
            out.append(("size_mismatch", ERROR, f"annotation size {aw}×{ah} ≠ image size {shown[0]}×{shown[1]}"))
        elif o != 1 and o not in ROTATED:
            out.append(("exif_present", WARN, f"EXIF orientation {o} ({ORIENT_TEXT.get(o, '?')}): the image size matches, "
                                              "but boxes may be mirrored / upside down in EXIF-aware tools"))
    elif o != 1:
        out.append(("exif_present", WARN, f"EXIF orientation {o} ({ORIENT_TEXT.get(o, '?')}): make sure the annotation "
                                          "tool and the trainer both apply it (or strip it)"))
    return out


# =============================================================================
# Low-priority metrics (information only)
# =============================================================================
def metrics(path) -> dict | None:
    try:
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im.draft("L", (512, 512))                     # JPEG: decode at reduced size (fast)
            im = ImageOps.exif_transpose(im).convert("L")
            im.thumbnail((512, 512))
            a = np.asarray(im, dtype=np.float32)
    except Exception:
        return None
    if a.size < 16:
        return None
    lap = a[1:-1, 1:-1] * -4 + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return {"blur": float(lap.var()), "mean": float(a.mean()), "std": float(a.std()),
            "dark": float((a < 30).mean()), "bright": float((a > 235).mean())}


def judge_metrics(m: dict | None, thr: dict | None = None) -> list[tuple]:
    if not m:
        return []
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    out = []
    if m["blur"] < t["blur"]:
        out.append(("blur", INFO, f"sharpness {m['blur']:.0f} (below {t['blur']:.0f})"))
    if m["mean"] < t["dark_mean"] or m["dark"] > t["dark_frac"]:
        out.append(("dark", INFO, f"mean brightness {m['mean']:.0f}, {m['dark'] * 100:.0f} % near black"))
    if m["mean"] > t["bright_mean"] or m["bright"] > t["bright_frac"]:
        out.append(("bright", INFO, f"mean brightness {m['mean']:.0f}, {m['bright'] * 100:.0f} % near white"))
    if m["std"] < t["low_std"]:
        out.append(("low_contrast", INFO, f"contrast {m['std']:.0f} (below {t['low_std']:.0f})"))
    return out
