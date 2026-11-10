# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Label colours that stay apart.

The old colours came from a hash of the label (md5 -> hue): two labels often landed a few degrees apart, so busy
images showed many near-identical outlines. Now:

  * a fixed palette, built once by farthest-point sampling in CIELAB (lightness weighted down: outlines are told
    apart by hue and saturation more than by lightness) over vivid, mid-to-light colours that read on photos — the
    first colours are as different as colours can be, every next one is the most different from all before it;
  * a label takes the first palette colour no other label of the dataset uses, in the order labels become known
    (the class list first: YOLO names / classes.txt), so the first 8–12 classes are maximally distinct;
  * the assignment is saved per dataset (.aio_qc/label_colors.json): a label keeps its colour across sessions,
    whatever order files are opened in.
"""
from __future__ import annotations

import colorsys
import json
import os
from functools import lru_cache

import numpy as np
from PyQt6.QtGui import QColor

MAX_COLORS = 256
FILE = "label_colors.json"
PALETTE = (                      # palette(build=True) at the time of writing; test_colors checks they agree
    "#ff261f", "#00ff00", "#1f69ff", "#00e6e6", "#ffd400", "#ff00b3", "#c7825a", "#7ec75a", "#bb00ff", "#67a2e6",
    "#c75a89", "#ff8800", "#ff0066", "#c867e6", "#00ffbb", "#b8e600", "#c7b85a", "#00ff77", "#ff7373", "#ff47f9",
    "#1fd2ff", "#50c700", "#1c7ae6", "#c78b00", "#c74c18", "#5ac79b", "#ff73d0", "#b4c738", "#8b47ff", "#ff4797",
    "#00c763", "#8567e6", "#ff5e00", "#ffc273", "#e61c3d", "#e68840", "#e6e600", "#b940e6", "#c7b018", "#e640ca",
    "#80ff00", "#e6406c", "#5ac77a", "#c75a68", "#00e636", "#ffb200", "#ff9373", "#5a77c7", "#00e6c7", "#9ee640",
    "#ff6047", "#00a2ff", "#ee00ff", "#38c741", "#00bbff", "#c74b38", "#a6c75a", "#c75ac0", "#ff739d", "#00ff99",
    "#00bac7", "#7385ff", "#ffc847", "#c75aa2", "#c7685a", "#ff1fd2", "#e6e167", "#e6e040", "#c79c38", "#ff0099",
    "#fa73ff", "#8ae600", "#c7975a", "#975ac7", "#ffab73", "#e6871c", "#ff47b6", "#67e6d0", "#00c78b", "#c76338",
    "#ff0080", "#b938c7", "#cb1ce6", "#ffda73", "#ff4766", "#78ff47", "#81e667", "#bfe640", "#ff73b4", "#e64b1c",
    "#e66300", "#1892c7", "#b473ff", "#c73892", "#ffaf47", "#92c738", "#ff8547", "#40e693", "#00e5ff", "#ff0033",
    "#739dff", "#c78938", "#ff73e8", "#187bc7", "#1f78ff", "#c738aa", "#5ac761", "#ff3c00", "#b647ff", "#c75d00",
    "#71c738", "#ffe047", "#b3c700", "#00c728", "#c718c7", "#e6b800", "#3de600", "#e61c6c", "#ff00ee", "#c79e18",
    "#7373ff", "#40e66c", "#ff7385", "#b7e667", "#73c2ff", "#ff004d", "#ffa51f", "#e61c22", "#00ff3c", "#67e6ab",
    "#38c7c2", "#e6404b", "#e64098", "#c77b18", "#ff47c8", "#d0e667", "#c7c238", "#c7af38", "#e66740", "#1c9be6",
    "#73abff", "#cfe600", "#e6407d", "#92c700", "#82e640", "#c7a65a", "#9ee667", "#4760ff", "#ff0000", "#c0c75a",
    "#1fff62", "#00ffd4", "#18c79e", "#1f96ff", "#e6cf00", "#00c7b3", "#a2e61c", "#6be600", "#e67067", "#677ce6",
    "#67e674", "#00ffaa", "#ff4747", "#b15ac7", "#c77b38", "#40e656", "#e60082", "#ff47e7", "#71c700", "#00e6b0",
    "#ff4b1f", "#5eff00", "#61e640", "#e6004d", "#67e689", "#e6a940", "#5a82c7", "#00c7e6", "#c75a77", "#d21fff",
    "#e6a91c", "#00b0e6", "#00ff88", "#5ac7aa", "#67e6c0", "#e68d67", "#408de6", "#e66781", "#c738bd", "#e6c067",
    "#e373ff", "#00e6d6", "#5a6cc7", "#e67300", "#e61c5f", "#ff00dd", "#c75ab1", "#e167e6", "#ff6e00", "#c7ba00",
    "#e600b8", "#af67e6", "#4785ff", "#ff9d47", "#ffc300", "#e600a1", "#1ce687", "#5ac789", "#e69840", "#e667af",
    "#e6002e", "#e60091", "#47f3ff", "#e6d567", "#e6a667", "#e6bf40", "#e640b4", "#67e69a", "#9873ff", "#38c776",
    "#c76a18", "#ff691f", "#e6951c", "#3889c7", "#676be6", "#c7c700", "#e6361c", "#ff961f", "#28c700", "#d540e6",
    "#c75a5a", "#ce47ff", "#e64640", "#e65f1c", "#e667cc", "#38c754", "#d5e640", "#e61f00", "#cc73ff", "#e6679a",
    "#e600d6", "#e66770", "#e64500", "#e67c67", "#94c75a", "#406ce6", "#59c738", "#ffd447", "#5aff47", "#e65640",
    "#ed47ff", "#de00e6", "#00e663", "#c78d5a", "#ffbc47", "#1ce6a2",
)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def palette(build: bool = False) -> tuple[str, ...]:
    """MAX_COLORS hex colours, most distinct first. The built table is stored above (no numpy work at runtime)."""
    return _build() if build else PALETTE


@lru_cache(maxsize=1)
def _build() -> tuple[str, ...]:
    cand = []
    for h in range(0, 360, 2):
        for s in (0.55, 0.72, 0.88, 1.0):
            for v in (0.78, 0.9, 1.0):
                cand.append(colorsys.hsv_to_rgb(h / 360, s, v))
    rgb = np.array(cand, float)
    lab = _srgb_to_lab(rgb)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (lab[:, 0] >= 48) & (lab[:, 0] <= 90) & (chroma >= 38)      # vivid, readable on photos
    rgb, lab = rgb[keep], lab[keep]
    w = lab * np.array([0.6, 1.0, 1.0])                                # lightness counts less for outlines
    first = int(np.argmin(np.abs(rgb - [1.0, 0.15, 0.15]).sum(1)))     # start from a strong red
    chosen = [first]
    d = np.linalg.norm(w - w[first], axis=1)
    for _ in range(min(MAX_COLORS, len(rgb)) - 1):
        n = int(np.argmax(d))
        chosen.append(n)
        d = np.minimum(d, np.linalg.norm(w - w[n], axis=1))
    return tuple(QColor(*(int(round(x * 255)) for x in rgb[i])).name() for i in chosen)


def next_color(used: set[str]) -> QColor:
    for hx in palette():
        if hx not in used:
            return QColor(hx)
    n = len(used)                                                      # > MAX_COLORS labels: still spread out
    r, g, b = colorsys.hsv_to_rgb((n * 0.61803398875) % 1.0, 0.65 + 0.3 * ((n // 7) % 2), 0.95)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def assign(label: str, label_dict: dict) -> None:
    """label_dict: label -> QColor (the app's). A new label gets the most distinct colour still free."""
    if label in label_dict:
        return
    label_dict[label] = next_color({c.name() for c in label_dict.values()})


def load(root, label_dict: dict) -> None:
    """The dataset's saved colours (label -> '#rrggbb') into label_dict."""
    try:
        with open(os.path.join(str(root), ".aio_qc", FILE), encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return
    for label, hx in (saved.get("colors") or {}).items() if isinstance(saved, dict) else ():
        c = QColor(str(hx))
        if c.isValid() and label not in label_dict:
            label_dict[label] = c


def save(root, label_dict: dict) -> bool:
    """Write the colours when they changed. -> written?"""
    folder = os.path.join(str(root), ".aio_qc")
    path = os.path.join(folder, FILE)
    data = {"version": 1, "colors": {k: v.name() for k, v in label_dict.items()}}
    try:
        with open(path, encoding="utf-8") as f:
            if json.load(f) == data:
                return False
    except (OSError, ValueError):
        pass
    try:
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
