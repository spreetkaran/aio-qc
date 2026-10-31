# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
View-time label mapping.

MappedAdapter wraps a dataset adapter: load() shows mapped labels, save() writes the
ORIGINAL labels back unless the user deliberately changed a label. Files are never
touched by the mapping itself.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path


class MappedAdapter:
    def __init__(self, base, mapping: dict[str, str] | None = None):
        self.base = base
        self.mapping = {k: v for k, v in (mapping or {}).items() if v and v != k}
        # reverse map for mapped names that come from exactly one source label
        rev: dict[str, list[str]] = {}
        for src, dst in self.mapping.items():
            rev.setdefault(dst, []).append(src)
        self.reverse_unique = {dst: srcs[0] for dst, srcs in rev.items() if len(srcs) == 1}

    def __getattr__(self, name):          # type, root, class_map, image_size, orphans, discover, ...
        return getattr(self.base, name)

    @property
    def active(self) -> bool:
        return bool(self.mapping)

    def map_label(self, label: str) -> str:
        return self.mapping.get(label, label)

    @property
    def class_names(self) -> list[str]:
        return list(dict.fromkeys(self.map_label(n) for n in self.base.class_names))

    def sample_for_path(self, path):
        return self.base.sample_for_path(path)

    def order_key(self, sample):
        return self.base.order_key(sample)

    def load(self, sample, image_size=None, light=False):
        ann = self.base.load(sample, image_size, light=light)
        if self.mapping and ann.tags:                   # classification: image-level classes
            ann.orig_tags = list(ann.tags)
            ann.tags = [self.mapping.get(t, t) for t in ann.tags]
        if self.mapping:
            for s in ann.shapes:
                lab = s.get("label")
                new = self.mapping.get(lab)
                if new is not None:
                    s["_orig_label"] = lab
                    s["label"] = new
        return ann

    def _restore(self, ann):
        if ann.orig_tags is not None:                   # unchanged mapped tags go back to their original names
            mapped = [self.mapping.get(t, t) for t in ann.orig_tags]
            ann.tags = list(ann.orig_tags) if ann.tags == mapped else ann.tags
            ann.orig_tags = None
        for s in ann.shapes:
            orig = s.pop("_orig_label", None)
            label = s.get("label")
            if s.get("_new") and label in self.reverse_unique:
                s["label"] = self.reverse_unique[label]         # new shape: write the dataset's own name
            elif orig is not None and label == self.mapping.get(orig):
                s["label"] = orig                      # untouched -> write original
            elif not self.base.can_use_label(label) and label in self.reverse_unique:
                s["label"] = self.reverse_unique[label]  # e.g. YOLO: mapped name -> real class

    def save(self, sample, ann):
        self._restore(ann)
        self.base.save(sample, ann)

    def can_use_label(self, label: str) -> bool:
        return self.base.can_use_label(label) or (
            label in self.reverse_unique and self.base.can_use_label(self.reverse_unique[label]))


# -----------------------------------------------------------------------------
# Import / export
# -----------------------------------------------------------------------------
def load_mapping_file(path: str | Path) -> dict[str, str]:
    """
    Accepts:
      * JSON  {"old": "new", ...}
      * Python file with a CLASS_MAP = {...} (+ optional LABEL_ALIASES) literal (e.g. config.py)
      * text  lines "old -> new" / "old: new" / "old,new"
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return {str(k): str(v) for k, v in data.items()}
    if path.suffix.lower() == ".py":
        tree = ast.parse(text)
        found: dict[str, dict] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("CLASS_MAP", "LABEL_ALIASES", "LABEL_MAP"):
                    try:
                        found[name] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
        if not found:
            raise ValueError("No CLASS_MAP / LABEL_MAP literal found in the Python file")
        cmap = dict(found.get("CLASS_MAP") or found.get("LABEL_MAP") or {})
        for alias, target in (found.get("LABEL_ALIASES") or {}).items():   # alias -> label -> canonical
            cmap.setdefault(alias, cmap.get(target, target))
        return {str(k): str(v) for k, v in cmap.items()}
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"\s*(?:->|=>|:|,|\t)\s*", line, maxsplit=1)
        if len(parts) == 2 and parts[0] and parts[1]:
            out[parts[0]] = parts[1]
    return out
