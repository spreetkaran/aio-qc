# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model-finding decisions that must survive a re-run (Qt-free): <root>/.aio_qc/model_rejects.json

A rejected finding ("the label is right, the model is wrong") is remembered by WHAT it is, not by its row:
    file (root-relative) | kind | label | model's label | box rounded to 1 % of the image
so the same finding is hidden on every later run — with any thresholds — until the file's labels change it away,
and it can be shown again (“Show rejected”) or forgotten (“Forget rejections”).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .io import atomic_write_text

NAME = "model_rejects.json"


def finding_key(root, key: str, f, W, H) -> str:
    try:
        rel = os.path.relpath(key, str(root)).replace("\\", "/")
    except ValueError:
        rel = key
    box = f.box if f.box is not None else f.gt_box
    if box is not None and W and H:
        b = (round(box[0] / W, 2), round(box[1] / H, 2), round(box[2] / W, 2), round(box[3] / H, 2))
    else:
        b = None
    return f"{rel}|{f.kind}|{f.gt_label}|{f.pred_label}|{b}"


class RejectLedger:
    def __init__(self, root):
        self.path = Path(root) / ".aio_qc" / NAME
        self.items: dict[str, dict] = {}
        try:
            self.items = json.loads(self.path.read_text(encoding="utf-8")).get("items", {})
        except (OSError, ValueError):
            pass

    def keys(self) -> set:
        return set(self.items)

    def add(self, keys, note: str = ""):
        now = time.strftime("%Y-%m-%d %H:%M")
        for k in keys:
            self.items[k] = {"when": now, "note": note}
        self.save()

    def remove(self, keys):
        for k in keys:
            self.items.pop(k, None)
        self.save()

    def clear(self):
        self.items = {}
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps({"version": 1, "items": self.items}, ensure_ascii=False,
                                                separators=(",", ":")))
