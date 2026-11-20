# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Building the request (Qt-free): the fixed part first — rules, classes, the answer's shape, the view's examples —
then the image to annotate. The fixed part is the same in every request of a view, so services that cache prompt
prefixes (LiteLLM / Gemini / OpenAI / Anthropic) charge it once.
"""
from __future__ import annotations

import fnmatch
import os
import re
import threading

import cv2

from AIO_QC.tools.frame_extractor.core.ai_annotate import profiles as P

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FMT_TEXT = {
    "yxyx_1000": "[ymin, xmin, ymax, xmax] as integers from 0 to 1000 relative to the image's height and width",
    "xyxy_1000": "[xmin, ymin, xmax, ymax] as integers from 0 to 1000 relative to the image's width and height",
    "xyxy_px": "[xmin, ymin, xmax, ymax] in pixels of the image as you see it ({w} × {h})",
    "xyxy_norm": "[xmin, ymin, xmax, ymax] as fractions from 0 to 1 of the image's width and height",
    "yxyx_norm": "[ymin, xmin, ymax, xmax] as fractions from 0 to 1 of the image's height and width",
}


def box_key(fmt):
    return "box_2d" if fmt == "yxyx_1000" else "bbox"


def encode(img, max_side, round_to=0, quality=92):
    """BGR image -> (jpeg bytes, sent_w, sent_h, scale_x, scale_y) — scale = original px per sent px."""
    h, w = img.shape[:2]
    s = min(1.0, float(max_side) / max(h, w)) if max_side else 1.0
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if round_to:
        nw, nh = max(round_to, nw // round_to * round_to), max(round_to, nh // round_to * round_to)
    small = img if (nw, nh) == (w, h) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the image")
    return buf.tobytes(), nw, nh, w / nw, h / nh


def from_pixels(box, fmt, w, h):
    """(x1, y1, x2, y2) in pixels of an image of w × h -> the model's own convention (for examples)."""
    x1, y1, x2, y2 = box
    if fmt == "yxyx_1000":
        return [round(y1 / h * 1000), round(x1 / w * 1000), round(y2 / h * 1000), round(x2 / w * 1000)]
    if fmt == "xyxy_1000":
        return [round(x1 / w * 1000), round(y1 / h * 1000), round(x2 / w * 1000), round(y2 / h * 1000)]
    if fmt == "xyxy_norm":
        return [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
    if fmt == "yxyx_norm":
        return [round(y1 / h, 4), round(x1 / w, 4), round(y2 / h, 4), round(x2 / w, 4)]
    return [round(x1), round(y1), round(x2), round(y2)]


# ------------------------------------------------------------------------------------------------ references
def known_boxes(img_path):
    """Boxes already drawn for an image: its LabelMe JSON, else its YOLO txt (+ classes.txt) — [(label, box px)]."""
    from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import read_labels
    return read_labels(img_path)


def expand(paths, base_dir):
    out = []
    for p in paths or []:
        full = p if os.path.isabs(p) else os.path.join(base_dir or ".", p)
        if os.path.isdir(full):
            for f in sorted(os.listdir(full)):
                if f.lower().endswith(IMAGE_EXTS) and not f.startswith("."):
                    out.append(os.path.join(full, f))
        elif os.path.isfile(full):
            out.append(full)
    return out


def view_for(cfg, image_path: str) -> str:
    """The view whose pattern fits the image (file name or any part of its path); else 'default'."""
    name, path = os.path.basename(image_path), image_path.replace("\\", "/")
    for k, v in (cfg["references"].get("views") or {}).items():
        pat = (v.get("match") or "").strip()
        if k == "default" or not pat:
            continue
        for p in [x.strip() for x in re.split(r"[;,]", pat) if x.strip()]:
            if p.startswith("/") and p.endswith("/") and len(p) > 2:
                if re.search(p[1:-1], path):
                    return k
            elif fnmatch.fnmatch(name, p) or fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, f"*{p}*"):
                return k
    return "default"


def draw(img, boxes):
    out = img.copy()
    lw = max(2, int(round(min(img.shape[:2]) / 300)))
    for label, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), lw)
        cv2.putText(out, label, (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, lw / 4),
                    (0, 0, 255), max(1, lw // 2), cv2.LINE_AA)
    return out


class References:
    """Encoded examples per view, made once per run (thread-safe)."""

    def __init__(self, cfg, base_dir=""):
        self.cfg, self.base = cfg, base_dir
        self._cache, self._lock = {}, threading.Lock()

    def parts(self, view: str):
        with self._lock:
            if view not in self._cache:
                self._cache[view] = self._build(view)
            return self._cache[view]

    def _build(self, view):
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        ref = self.cfg["references"]
        v = (ref.get("views") or {}).get(view) or {}
        paths = expand(v.get("images"), self.base)
        if view != "default" and not paths:
            paths = expand((ref.get("views") or {}).get("default", {}).get("images"), self.base)
        fmt = P.box_format(self.cfg)
        out = []
        for i, p in enumerate(paths, 1):
            img = imread(p)
            if img is None:
                continue
            boxes = known_boxes(p)
            h, w = img.shape[:2]
            shown = draw(img, boxes) if boxes and ref.get("draw_boxes", True) else img
            data, sw, sh, _, _ = encode(shown, int(ref.get("max_side") or 1024))
            if boxes:
                listed = "; ".join(f"{lab} {from_pixels(b, fmt, w, h) if fmt != 'xyxy_px' else from_pixels((b[0] * sw / w, b[1] * sh / h, b[2] * sw / w, b[3] * sh / h), fmt, sw, sh)}"
                                   for lab, b in boxes)
                cap = f"EXAMPLE {i}: correctly annotated — {listed}."
            else:
                cap = f"EXAMPLE {i}: an image of this scene."
            out.append(("text", cap))
            out.append(("image", data, "image/jpeg"))
        return out


# ------------------------------------------------------------------------------------------------ prompts
def system_text(cfg, sent_w=0, sent_h=0) -> str:
    mode = cfg["mode"]
    fmt = P.box_format(cfg)
    bk = box_key(fmt)
    lines = ["You are a meticulous image annotator preparing training data for a computer-vision model.",
             "Be precise and conservative: annotate only what is clearly visible; never invent objects."]
    if cfg.get("project"):
        lines.append(f"Project: {cfg['project']}." + (f" {cfg['description']}" if cfg.get("description") else ""))
    if cfg["classes"] and mode != "classify":
        lines.append("\nCLASSES (use these labels exactly):")
        for c in cfg["classes"]:
            extra = []
            if c.get("inside"):
                extra.append(f"only inside a {c['inside']}")
            if c.get("text"):
                extra.append("also read its text exactly, character by character, into \"text\"")
            if c.get("parts"):
                extra.append(f"also box its parts {', '.join(c['parts'])} in \"parts\"")
            if c.get("max_per_image"):
                extra.append(f"at most {c['max_per_image']} per image")
            lines.append(f"- {c['name']}: {(c.get('description') or '').strip() or c['name']}"
                         + (f" ({'; '.join(extra)})" if extra else ""))
    if cfg["flags"]:
        lines.append("\nQUESTIONS about the whole image (answer true or false in \"flags\"):")
        lines += [f"- {f['name']}: {f.get('description') or f['name']}" for f in cfg["flags"]]
    fields = cfg["checks"].get("fields") or {}
    if fields:
        lines.append("\nALSO REPORT (top-level fields):")
        lines += [f"- {k}: {v}" for k, v in fields.items()]
    coords = FMT_TEXT[fmt].format(w=sent_w or "W", h=sent_h or "H")
    shape = {"detections": [{"label": "<class>", bk: "[4 numbers]", "confidence": "0.0-1.0"}]}
    if any(c.get("text") for c in cfg["classes"]):
        shape["detections"][0]["text"] = "<text read, if the class reads text>"
    if any(c.get("parts") for c in cfg["classes"]):
        shape["detections"][0]["parts"] = {"<part class>": "[4 numbers]"}
    if mode == "classify":
        shape = {}
    if mode == "detect_vlm":
        shape = {"boxes": [{"id": 1, "keep": "true/false", "label": "<class>", "confidence": "0.0-1.0"}]}
        if any(c.get("text") for c in cfg["classes"]):
            shape["boxes"][0]["text"] = "<text read, if the class reads text>"
    if cfg["flags"]:
        shape["flags"] = {f["name"]: "true/false" for f in cfg["flags"]}
    for k in fields:
        shape[k] = "…"
    import json
    lines.append("\nANSWER with ONLY one JSON object, no other text, shaped like:\n" + json.dumps(shape, indent=1))
    if mode == "vlm_detect":
        lines.append(f"\"{bk}\" is {coords} of the IMAGE TO ANNOTATE (the last image). One tight box per object; "
                     "include an object only when you are reasonably sure. An empty \"detections\" list is a valid "
                     "answer.")
        if any(c.get("parts") for c in cfg["classes"]):
            lines.append("Part boxes use the same coordinates.")
    elif mode == "detect_vlm":
        lines.append("The IMAGE TO ANNOTATE shows numbered candidate boxes found by a detector. For EVERY numbered "
                     "box say whether it is a correct, tight box of one of the classes (keep), and its correct "
                     "label. Reject boxes on the wrong object, duplicates and boxes that cut the object badly.")
    if (cfg["vlm"].get("instructions") or "").strip():
        lines.append("\nPROJECT RULES:\n" + cfg["vlm"]["instructions"].strip())
    return "\n".join(lines)


def image_intro(cfg, sent_w, sent_h, crop=None, view_note="") -> str:
    t = f"IMAGE TO ANNOTATE ({sent_w} × {sent_h} pixels)"
    if crop:
        t += f" — a close-up crop of a {crop}; annotate inside it"
    if view_note:
        t += f". Camera: {view_note}"
    return t + ":"
