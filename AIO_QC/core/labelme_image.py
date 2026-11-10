# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
LabelMe `imageData`: the image embedded in the JSON as base64. The key is the same in labelme 3 … 6
("imageData", with "imagePath" / "imageHeight" / "imageWidth"); labelme 6 requires the key to be present (null = not
embedded), so AIO QC always writes it — the base64 string or null.

How each labelme version produces the embedded bytes (so AIO QC writes what that version would have written):
    5.x and older   PIL open -> EXIF orientation -> re-encode (JPEG for .jpg/.jpeg, otherwise PNG)
    6.x             the file's own bytes when no EXIF rotation is needed and it is a JPEG/PNG,
                    otherwise re-encode (PNG when the image has alpha, else JPEG quality 95)
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

try:
    from PIL import Image  # type: ignore
except Exception:                                        # pragma: no cover - PIL is optional
    Image = None

KEY = "imageData"
_ORIENTATION = 0x0112


def major(version) -> int:
    try:
        return int(str(version or "5").strip().lstrip("v").split(".")[0])
    except ValueError:
        return 5


def _oriented(im):
    try:
        o = (im.getexif() or {}).get(_ORIENTATION, 1)
    except Exception:
        o = 1
    if o in (None, 1):
        return im
    from PIL import ImageOps  # type: ignore
    return ImageOps.exif_transpose(im)


def encode_image_bytes(image_path, version="6") -> bytes:
    """Bytes labelme `version` would embed for this image file."""
    p = Path(image_path)
    raw = p.read_bytes()
    if Image is None:
        return raw
    ext = p.suffix.lower()
    im = Image.open(io.BytesIO(raw))
    oriented = _oriented(im)
    if major(version) >= 6:
        if oriented is im and ext in (".jpg", ".jpeg", ".png"):
            return raw
        fmt = "PNG" if "A" in oriented.mode else "JPEG"
        out = io.BytesIO()
        (oriented if fmt == "PNG" or oriented.mode in ("RGB", "L") else oriented.convert("RGB")).save(
            out, format=fmt, quality=95)
        return out.getvalue()
    fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
    out = io.BytesIO()
    img = oriented if fmt == "PNG" or oriented.mode in ("RGB", "L") else oriented.convert("RGB")
    img.save(out, format=fmt)
    return out.getvalue()


def encode_b64(image_path, version="6") -> str:
    return base64.b64encode(encode_image_bytes(image_path, version)).decode("ascii")


def sniff_ext(data: bytes) -> str:
    """File extension for decoded image bytes (to give the editor a real image file)."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".png"


def image_file_of(json_path, data: dict) -> Path | None:
    """The image file a LabelMe JSON points to (imagePath relative to the JSON, Windows separators accepted)."""
    ip = data.get("imagePath")
    if not ip:
        return None
    p = Path(json_path).parent / str(ip).replace("\\", "/")
    return p if p.is_file() else None
