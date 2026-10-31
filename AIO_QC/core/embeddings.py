# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Visual-similarity engine, app side (Qt-free, numpy only).

Models   a model is a FOLDER: a Hugging Face snapshot (config.json — DINOv3 / DINOv2 / SigLIP / SigLIP2 / CLIP), a
         timm or open_clip folder, or the DINOv3 repo + .pth. Zips are extracted once (streamed, resumable: an
         interrupted extraction is redone, a finished one is reused) into a models folder the user chooses.
Cache    vectors are kept per model in <root>/.aio_qc/embeddings/<model id>.npz (float16) under an id that changes
         only when the object's image or geometry changes — relabelling never forces a re-embed. Stale ids are
         dropped on save, so the cache never grows past the dataset.
Analysis from L2-normalised vectors + their k nearest neighbours:
         likely mislabel   most look-alikes carry ONE other label, own label (almost) absent among them
         atypical          far from its own label's typical look (robust z-score vs the label centre)
         look-alike pair   near-identical objects with different labels
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Downloadable presets: (title, Hugging Face repo, approx. download size, note)
PRESETS = [
    ("DINOv3 ViT-S/16 — fast, strong look-alike features", "facebook/dinov3-vits16-pretrain-lvd1689m", "≈ 90 MB",
     "gated: accept the licence on huggingface.co and use a token"),
    ("DINOv3 ViT-L/16 — best quality, slower", "facebook/dinov3-vitl16-pretrain-lvd1689m", "≈ 1.2 GB",
     "gated: accept the licence on huggingface.co and use a token"),
    ("DINOv2 ViT-S/14 — fast, no licence gate", "facebook/dinov2-small", "≈ 90 MB", ""),
    ("SigLIP2 Large patch16-384 — semantic, text-aware", "google/siglip2-large-patch16-384", "≈ 3.5 GB", ""),
    ("CLIP ViT-B/32 — small, semantic", "openai/clip-vit-base-patch32", "≈ 600 MB", ""),
    ("CLIP ViT-L/14 — semantic, larger", "openai/clip-vit-large-patch14", "≈ 1.7 GB", ""),
]
MARKERS = ("config.json", "open_clip_config.json", "hubconf.py")


# ---------------------------------------------------------------------- model folders
def find_model_root(path) -> Path | None:
    """The folder that holds the model (a zip often wraps it in one or two extra folders)."""
    path = Path(path)
    if not path.is_dir():
        return None
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        if any(m in files for m in MARKERS):
            return Path(root)
    return None


def model_kind(root) -> str:
    root = Path(root)
    if (root / "hubconf.py").is_file():
        return "DINOv3 repo (torch.hub)"
    if (root / "open_clip_config.json").is_file():
        return "open_clip"
    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    if "architecture" in cfg and "model_type" not in cfg:
        return f"timm · {cfg['architecture']}"
    return f"transformers · {cfg.get('model_type', '?')}"


def folder_size(root) -> int:
    total = 0
    for r, _d, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total


def model_id(root) -> str:
    """Stable id of a model's weights (path + weight files' size / mtime) -> cache file name."""
    root = Path(root)
    h = hashlib.sha1(str(root.resolve()).encode())
    for r, _d, files in os.walk(root):
        for f in sorted(files):
            if f.endswith((".safetensors", ".bin", ".pth", ".pt")):
                st = os.stat(os.path.join(r, f))
                h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def extract_zip(zip_path, dest_dir, progress=None, cancel=None) -> Path:
    """Extract (streamed) into dest_dir/<zip name>/ once; a finished extraction is reused. -> the model root."""
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    target = dest_dir / zip_path.stem
    marker = target / ".aioqc_extracted"
    sig = f"{zip_path.stat().st_size}:{int(zip_path.stat().st_mtime)}"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == sig:
        root = find_model_root(target)
        if root is not None:
            return root
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        total = sum(m.file_size for m in members) or 1
        done = 0
        for m in members:
            if cancel is not None and cancel():
                raise InterruptedError("cancelled")
            name = Path(m.filename)
            if name.is_absolute() or ".." in name.parts:            # zip-slip guard
                continue
            out = target / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                while True:
                    chunk = src.read(8 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    marker.write_text(sig, encoding="utf-8")
    root = find_model_root(target)
    if root is None:
        raise ValueError(f"{zip_path.name}: no model inside (expected config.json, open_clip_config.json or hubconf.py)")
    return root


# ---------------------------------------------------------------------- cache
class EmbeddingCache:
    def __init__(self, root, mid: str, mode: str):
        self.path = Path(root) / ".aio_qc" / "embeddings" / f"{mid}_{mode}.npz"
        self.index: dict[str, int] = {}
        self.vecs = None
        try:
            with np.load(self.path, allow_pickle=False) as z:
                ids = z["ids"]
                self.vecs = z["vecs"]
            self.index = {str(k): i for i, k in enumerate(ids)}
        except (OSError, ValueError, KeyError):
            self.index, self.vecs = {}, None

    def get(self, ids: list[str]) -> tuple[np.ndarray, list[int]]:
        """-> (row per id or -1, positions of ids that are missing)"""
        rows = np.fromiter((self.index.get(i, -1) for i in ids), dtype=np.int64, count=len(ids))
        return rows, [p for p, r in enumerate(rows) if r < 0]

    def save(self, ids: list[str], vecs: np.ndarray):
        """Keep exactly these ids (the current dataset) — stale entries go."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.asarray(ids, dtype=str), vecs=vecs.astype(np.float16))
        os.replace(tmp, self.path)
        self.index = {k: i for i, k in enumerate(ids)}
        self.vecs = vecs


def geom_id(shape: dict) -> str:
    pts = shape.get("points") or []
    return (shape.get("shape_type") or "polygon") + ":" + ",".join(f"{float(p[0]):.2f}:{float(p[1]):.2f}" for p in pts)


def file_sig(path) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except (OSError, TypeError):
        return "?"


# ---------------------------------------------------------------------- analysis
@dataclass
class SimFinding:
    i: int
    kind: str              # mislabel | atypical | lookalike
    label: str
    suggest: str
    score: float
    evidence: str
    other: int = -1        # the look-alike's item index


KINDS = {"mislabel": "Likely mislabel", "atypical": "Atypical for its label", "lookalike": "Look-alike, other label"}


def analyze(vecs: np.ndarray, labels: list[str], nn_idx: np.ndarray, nn_sim: np.ndarray, vote: float = 0.6,
            min_sim: float = 0.0, z_limit: float = 3.5, dup_sim: float = 0.97, min_class: int = 10,
            valid: np.ndarray | None = None) -> list[SimFinding]:
    n = len(labels)
    if n < 3:
        return []
    names = sorted(set(labels))
    code = {l: i for i, l in enumerate(names)}
    lab = np.fromiter((code[l] for l in labels), dtype=np.int32, count=n)
    valid = np.ones(n, bool) if valid is None else valid
    sizes = np.bincount(lab, minlength=len(names))
    sim = nn_sim.astype(np.float32)
    nl = lab[nn_idx]
    out: list[SimFinding] = []
    flagged: set[int] = set()
    # similarity scales differ per model (DINO ≈ 0.3–0.8, CLIP ≈ 0.7–0.95): a vote counts only when the neighbours
    # are about as close as neighbours usually are in THIS run
    floor = max(min_sim, 0.6 * float(np.median(sim[valid, 0]))) if valid.any() else min_sim
    # ---- likely mislabels: weighted label vote of the k nearest neighbours
    w = np.clip(sim, 0, None)
    for i in np.nonzero(valid)[0]:
        ok = valid[nn_idx[i]]
        if ok.sum() < 3:
            continue
        ww, ll = w[i][ok], nl[i][ok]
        tot = ww.sum()
        if tot <= 0 or ww.mean() < floor:
            continue
        votes = np.bincount(ll, weights=ww, minlength=len(names)) / tot
        own = votes[lab[i]]
        votes[lab[i]] = -1
        b = int(np.argmax(votes))
        if votes[b] >= vote and own <= (1 - vote) / 2:
            cnt = int((ll == b).sum())
            rare = sizes[lab[i]] < len(nn_idx[i])                    # tiny own class: look-alikes can't share it
            flagged.add(int(i))
            out.append(SimFinding(int(i), "mislabel", labels[i], names[b], float(votes[b] * ww.mean()),
                                  f"{cnt} of its {len(ll)} closest look-alikes are '{names[b]}', "
                                  f"{int((ll == lab[i]).sum())} are '{labels[i]}'" +
                                  (f" (only {sizes[lab[i]]} '{labels[i]}' in the run — check carefully)" if rare else "")))
    # ---- atypical: robust z of the similarity to the label centre
    v = vecs.astype(np.float32)
    for c, name in enumerate(names):
        idx = np.nonzero((lab == c) & valid)[0]
        if len(idx) < min_class:
            continue
        cen = v[idx].mean(0)
        cen /= max(np.linalg.norm(cen), 1e-9)
        s = v[idx] @ cen
        med = np.median(s)
        mad = np.median(np.abs(s - med)) * 1.4826 or 1e-6
        z = (s - med) / mad
        for j in np.nonzero(z < -z_limit)[0]:
            if int(idx[j]) in flagged:
                continue
            flagged.add(int(idx[j]))
            out.append(SimFinding(int(idx[j]), "atypical", name, "", float(-z[j]),
                                  f"looks unlike other '{name}' (similarity to the typical '{name}' {s[j]:.2f}, "
                                  f"usual {med:.2f}; {abs(z[j]):.1f} robust SDs below)"))
    # ---- look-alike pairs with different labels: MUTUAL nearest neighbours only (one stray item cannot spawn
    # a row for every object it resembles); reported on the side whose label its neighbourhood disagrees with
    seen = set()
    first, fsim = nn_idx[:, 0], sim[:, 0]

    def agree(x):
        ok = valid[nn_idx[x]]
        return float((nl[x][ok] == lab[x]).mean()) if ok.any() else 0.0
    for i in np.nonzero(valid & (fsim >= dup_sim))[0]:
        j = int(first[i])
        if not valid[j] or lab[i] == lab[j] or int(first[j]) != int(i):
            continue
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        a, b = (int(i), j) if agree(int(i)) <= agree(j) else (j, int(i))
        if a in flagged:
            continue
        flagged.add(a)
        out.append(SimFinding(a, "lookalike", labels[a], labels[b], float(fsim[i]),
                              f"almost identical (similarity {fsim[i]:.3f}) to an object labelled '{labels[b]}'",
                              other=b))
    return out


# ---------------------------------------------------------------------- orderings (Crop Gallery)
def typicality(vecs: np.ndarray) -> np.ndarray:
    """Similarity of each vector to the group's centre (low = atypical)."""
    v = vecs.astype(np.float32)
    c = v.mean(0)
    c /= max(np.linalg.norm(c), 1e-9)
    return v @ c


def look_order(vecs: np.ndarray, seed: int = 0) -> np.ndarray:
    """Order that puts look-alikes next to each other: k-means (k ≈ √(n/20)), clusters largest first, inside each
    cluster most typical first. Pure numpy, O(n·k·iterations)."""
    n = len(vecs)
    if n <= 2:
        return np.arange(n)
    v = vecs.astype(np.float32)
    k = int(min(64, max(2, round((n / 20) ** 0.5))))
    rng = np.random.default_rng(seed)
    cent = v[rng.choice(n, k, replace=False)]
    for _ in range(12):
        a = np.argmax(v @ cent.T, axis=1)
        for c in range(k):
            m = a == c
            if m.any():
                x = v[m].mean(0)
                cent[c] = x / max(np.linalg.norm(x), 1e-9)
    a = np.argmax(v @ cent.T, axis=1)
    s = np.einsum("ij,ij->i", v, cent[a])
    size = Counter(a.tolist())
    rank = {c: r for r, (c, _n) in enumerate(size.most_common())}
    return np.lexsort((-s, np.fromiter((rank[c] for c in a), dtype=np.int64, count=n)))
