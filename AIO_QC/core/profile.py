# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image-level profiling and hashing (ported from dataset_analysis_v2.py, optimised):

- reduced-resolution decode (cv2.IMREAD_REDUCED_GRAYSCALE_*): JPEG is decoded at 1/2..1/8 scale
  directly in the DCT domain -> several times faster than full decode for lighting + pHash
- near-duplicate search via multi-index hashing (pigeonhole): O(n) buckets instead of O(n^2) pairs
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

ASPECT_RATIO_TOLERANCE = 0.035
SMALL_DIMENSION_THRESHOLD = 320
LOW_LIGHT_MEAN_GRAY = 70
VERY_LOW_LIGHT_MEAN_GRAY = 45
LOW_LIGHT_DARK_PIXEL_RATIO = 0.55
PHASH_SIZE = 16                     # 16x16 = 256-bit hash (same as the pipeline)
COMMON_RATIOS = {"1:1": 1.0, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9, "18.5:9": 18.5 / 9,
                 "19.5:9": 19.5 / 9, "20:9": 20 / 9, "21:9": 21 / 9}


def aspect_family(w: int, h: int) -> str:
    """Orientation-independent ratio family ('4:3' covers 3:4 too)."""
    if not w or not h:
        return "unknown"
    actual = max(w, h) / min(w, h)
    best, err = min(((n, abs(actual - t) / t) for n, t in COMMON_RATIOS.items()), key=lambda x: x[1])
    return best if err <= ASPECT_RATIO_TOLERANCE else f"other ({actual:.2f}:1)"


def orientation(w: int, h: int) -> str:
    return "square" if w == h else ("landscape" if w > h else "portrait")


def read_gray_reduced(path: str | Path, target: int = 256):
    """Grayscale decode at reduced scale; picks the largest reduction keeping >= target px."""
    if cv2 is None:
        return None
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is not None and min(img.shape[:2]) >= target:
        return img
    for flag in (cv2.IMREAD_REDUCED_GRAYSCALE_4, cv2.IMREAD_REDUCED_GRAYSCALE_2, cv2.IMREAD_GRAYSCALE):
        img = cv2.imdecode(buf, flag)
        if img is not None and (min(img.shape[:2]) >= target or flag == cv2.IMREAD_GRAYSCALE):
            return img
    return img


def lighting(gray) -> tuple[str, float, float]:
    mean = float(gray.mean())
    dark = float(np.mean(gray < 50))
    if mean < VERY_LOW_LIGHT_MEAN_GRAY or dark >= 0.75:
        cat = "very_lowlight"
    elif mean < LOW_LIGHT_MEAN_GRAY or dark >= LOW_LIGHT_DARK_PIXEL_RATIO:
        cat = "lowlight"
    else:
        cat = "well_lit"
    return cat, mean, dark


def phash(gray, size: int = PHASH_SIZE) -> np.ndarray:
    """DCT perceptual hash -> packed uint8 bits (size*size/8 bytes)."""
    small = cv2.resize(gray, (size * 4, size * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:size, :size]
    bits = (d > np.median(d.ravel()[1:])).ravel()
    return np.packbits(bits)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POP[np.bitwise_xor(a, b)].sum())


def near_duplicate_pairs(hashes: list[np.ndarray], threshold: int) -> list[tuple[int, int, int]]:
    """
    All pairs with Hamming distance <= threshold.
    Multi-index hashing: split each hash into (threshold + 1) chunks; two hashes within
    `threshold` bits must agree exactly on at least one chunk, so only bucket-mates are compared.
    """
    if not hashes:
        return []
    bits = [np.unpackbits(h) for h in hashes]
    nbits = len(bits[0])
    k = threshold + 1
    bounds = np.linspace(0, nbits, k + 1).astype(int)
    seen, out = set(), []
    for c in range(k):
        lo, hi = bounds[c], bounds[c + 1]
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, b in enumerate(bits):
            buckets[np.packbits(b[lo:hi]).tobytes() + bytes([hi - lo])].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    d = hamming(hashes[i], hashes[j])
                    if d <= threshold:
                        out.append((i, j, d))
    return out


def union_find_groups(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, *_ in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]
