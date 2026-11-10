# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Small annotated images for reports and hand-off hints (Qt-free; OpenCV, Pillow as a fallback).

    jpeg = render(path, boxes=[(x1, y1, x2, y2, "label", (r, g, b))], max_side=360)   -> bytes | None
Boxes are image pixels, or normalised 0..1 (YOLO) — detected per box. Unicode paths work on Windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _palette(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def render(path, boxes=(), max_side: int = 360, quality: int = 82, label_text: bool = True) -> bytes | None:
    if cv2 is not None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            img = None
        if img is None:
            return None
        H, W = img.shape[:2]
        s = min(1.0, max_side / max(W, H))
        if s < 1:
            img = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        lw = max(1, int(round(max(img.shape[:2]) / 220)))
        for b in boxes:
            x1, y1, x2, y2 = _px(b[:4], W, H)
            color = b[5] if len(b) > 5 and b[5] else (255, 64, 160)
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            p1 = (int(min(x1, x2) * s), int(min(y1, y2) * s))
            p2 = (int(max(x1, x2) * s), int(max(y1, y2) * s))
            cv2.rectangle(img, p1, p2, bgr, lw, cv2.LINE_AA)
            if label_text and len(b) > 4 and b[4]:
                fs = max(0.32, max(img.shape[:2]) / 900)
                cv2.putText(img, str(b[4])[:24], (p1[0] + 1, max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, fs, bgr,
                            1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
    try:                                                     # Pillow fallback
        import io

        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            s = min(1.0, max_side / max(W, H))
            if s < 1:
                im = im.resize((max(1, int(W * s)), max(1, int(H * s))))
            d = ImageDraw.Draw(im)
            for b in boxes:
                x1, y1, x2, y2 = _px(b[:4], W, H)
                color = tuple(b[5]) if len(b) > 5 and b[5] else (255, 64, 160)
                d.rectangle([min(x1, x2) * s, min(y1, y2) * s, max(x1, x2) * s, max(y1, y2) * s], outline=color,
                            width=max(1, int(max(im.size) / 220)))
                if label_text and len(b) > 4 and b[4]:
                    d.text((min(x1, x2) * s + 2, min(y1, y2) * s + 2), str(b[4])[:24], fill=color)
            out = io.BytesIO()
            im.save(out, "JPEG", quality=quality)
            return out.getvalue()
    except Exception:
        return None


def _px(v4, W, H):
    v = [float(x) for x in v4]
    if all(0 <= x <= 1.0001 for x in v):
        return v[0] * W, v[1] * H, v[2] * W, v[3] * H
    return v


def shape_boxes(ann, colors: dict | None = None) -> list:
    """Annotation -> [(x1, y1, x2, y2, label, rgb)] (pixels, or normalised for YOLO)."""
    out = []
    colors = colors if colors is not None else {}
    for sh in getattr(ann, "shapes", None) or []:
        pts = sh.get("points") or []
        if not pts:
            continue
        lab = str(sh.get("label"))
        if lab not in colors:
            colors[lab] = _palette(len(colors))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if sh.get("shape_type") == "circle" and len(pts) == 2:
            r = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
            xs, ys = [pts[0][0] - r, pts[0][0] + r], [pts[0][1] - r, pts[0][1] + r]
        out.append((min(xs), min(ys), max(xs), max(ys), lab, colors[lab]))
    return out


def save(path, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p
