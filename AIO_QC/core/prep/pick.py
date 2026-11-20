# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Pick images to label next (Qt-free): from a pool of unlabelled images choose the set that teaches the model the
most per image labelled — as varied as possible (no near-identical frames), optionally unlike what is already
labelled, optionally where the current model is unsure.

Features (one vector per image, L2-normalised):
    built-in   fast, no model: 16×16 grey layout + HSV colour histogram + 4×4 grid of edge-orientation histograms
               (decoded at reduced size, EXIF applied) — tells scenes, viewpoints, lighting and clutter apart
    model      DINOv3 / SigLIP2 / CLIP image embeddings from the Visual Similarity setup (the UI passes them in)

Selection — greedy k-center (farthest-point sampling): each pick is the image farthest from everything picked so
far (and from the already-labelled images, if given), so the set covers the whole variety of the pool instead of
the most common scene. With model uncertainty the distance is weighted by (1 + w · uncertainty), so among equally
new images the ones the model struggles with come first. The order of the picks is a ranking: label from the top.

Output: the picked images copied (or moved) into a folder with the same sub-folders, plus picked.csv (rank,
image, novelty, uncertainty). Logged and undoable like every Data Prep run.
"""
from __future__ import annotations

import csv
import os
import shutil

from . import common as C
from AIO_QC.core.diag import ignored as _ignored

GRID, HIST_BINS = 4, 8


def features(path, size: int = 64):
    """Built-in feature vector (float32, L2-normalised) or None if unreadable."""
    import numpy as np
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as im:
            im.draft("RGB", (size * 2, size * 2))          # JPEG: decode at 1/2 … 1/8 size (fast)
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = im.resize((size, size), Image.BILINEAR)
            rgb = np.asarray(im, dtype=np.float32) / 255.0
            hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    except Exception as _exc:
        _ignored(_exc)
        return None
    grey = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    # 1. layout: 16×16 grey, zero-mean / unit-norm (robust to overall brightness)
    g16 = grey.reshape(16, size // 16, 16, size // 16).mean(axis=(1, 3)).ravel()
    g16 = g16 - g16.mean()
    # 2. colour: HSV histogram 8×4×4 (hue weighted by saturation implicitly through the bins)
    h = np.minimum((hsv[..., 0] * 8).astype(int), 7)
    s = np.minimum((hsv[..., 1] * 4).astype(int), 3)
    v = np.minimum((hsv[..., 2] * 4).astype(int), 3)
    col = np.bincount((h * 16 + s * 4 + v).ravel(), minlength=128).astype(np.float32)
    col = np.sqrt(col / col.sum())
    # 3. structure: edge orientations on a 4×4 grid, 8 bins, magnitude weighted
    gy, gx = np.gradient(grey)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi
    b = np.minimum((ang * HIST_BINS).astype(int), HIST_BINS - 1)
    cell = size // GRID
    hog = np.zeros((GRID, GRID, HIST_BINS), np.float32)
    for i in range(GRID):
        for j in range(GRID):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            hog[i, j] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=HIST_BINS)
    hog = np.sqrt(hog.ravel())

    def unit(x):
        n = float(np.linalg.norm(x))
        return x / n if n > 1e-9 else x
    vec = np.concatenate([unit(g16) * 1.0, unit(col) * 0.8, unit(hog) * 1.0]).astype(np.float32)
    return unit(vec)


def feature_matrix(paths, progress=None, cancel=None, label="Reading images"):
    """(matrix (n, d) float32, ok mask) — unreadable images get a zero row and ok=False."""
    import numpy as np
    from .tasks import _pmap
    vecs = _pmap(features, list(paths), progress, label, cancel)
    d = next((len(v) for v in vecs if v is not None), 1)
    X = np.zeros((len(vecs), d), np.float32)
    ok = np.zeros(len(vecs), bool)
    for i, v in enumerate(vecs):
        if v is not None:
            X[i], ok[i] = v, True
    return X, ok


def _unit_rows(X):
    import numpy as np
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def k_center(X, k: int, seeds=None, weight=None, ok=None, chunk: int = 256, cancel=None, progress=None):
    """Greedy farthest-point selection. X: (n, d) rows L2-normalised. seeds: (m, d) already-labelled images.
    weight: (n,) ≥ 0 multiplier (1 = neutral). -> (picked indices in order, novelty of each pick, final distances)."""
    import numpy as np
    X = _unit_rows(np.asarray(X, np.float32))
    n = len(X)
    k = max(0, min(k, n if ok is None else int(ok.sum())))
    mind = np.full(n, np.inf, np.float32)
    if seeds is not None and len(seeds):
        S = _unit_rows(np.asarray(seeds, np.float32))
        for a in range(0, len(S), chunk):
            sim = X @ S[a:a + chunk].T
            mind = np.minimum(mind, np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim.max(axis=1))))
    w = np.ones(n, np.float32) if weight is None else np.asarray(weight, np.float32)
    valid = np.ones(n, bool) if ok is None else np.array(ok, bool)          # a copy: the caller's mask stays
    picked, novelty = [], []
    if not np.isfinite(mind).any():                      # no seeds: start with the most atypical image
        mean = X[valid].mean(axis=0) if valid.any() else X.mean(axis=0)
        mind = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X @ _unit_rows(mean[None])[0]))).astype(np.float32)
        first = True
    else:
        first = False
    for t in range(k):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        score = np.where(valid, mind * w, -1.0)
        i = int(np.argmax(score))
        if score[i] < 0:
            break
        picked.append(i)
        novelty.append(float(mind[i]) if not (first and t == 0) else float("nan"))
        valid[i] = False
        d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X @ X[i]))).astype(np.float32)
        mind = d if (first and t == 0) else np.minimum(mind, d)
        if progress is not None and (t % 50 == 0 or t == k - 1):
            progress(t + 1, k, "Choosing")
    return picked, novelty, mind


def pool(src, skip_labelled: bool = True, cancel=None) -> tuple[list[str], int]:
    """Images of the pool (sorted) and how many were skipped because they already have a label file."""
    images, jsons, _z, _o, _s = C.scan([src], cancel=cancel)
    if not skip_labelled:
        return images, 0
    have = {C.norm(os.path.splitext(j)[0]) for j in jsons}
    out = []
    skipped = 0
    for p in images:
        base = os.path.splitext(p)[0]
        txt = None
        q = p.replace("\\", "/")
        i = q.rfind("/images/")
        if i >= 0:
            txt = os.path.splitext(q[:i] + "/labels/" + q[i + 8:])[0] + ".txt"
        if C.norm(base) in have or (txt and os.path.isfile(txt)):
            skipped += 1
        else:
            out.append(p)
    return out, skipped


def plan(src, out, picked_paths, scores, copy=True, folders=None):
    """picked_paths in rank order; scores: [(novelty, uncertainty)] per pick. folders: a sub-folder per pick (a
    classifier's best guess: the picks arrive sorted into class folders, to be confirmed)."""
    from .tasks import Plan
    src, out = os.path.abspath(src), os.path.abspath(out)
    if C.inside(out, src):
        raise ValueError("Choose an output folder outside the pool folder.")
    P = Plan("pick", opts={"src": src, "out": out, "copy": copy, "scores": scores})
    dsts = []
    for i, p in enumerate(picked_paths):
        sub = folders[i] if folders and i < len(folders) and folders[i] else None
        dsts.append(os.path.join(out, _safe_dir(sub), os.path.basename(p)) if sub else
                    os.path.join(out, os.path.relpath(p, src)))
    dsts, renamed = _unique_names(src, out, picked_paths, dsts)
    for p, dst in zip(picked_paths, dsts):
        P.actions.append(("pick", p, dst, {}))
    if renamed:
        P.details["Renamed (the name was taken in that folder)"] = renamed[:1000]
    return P


def _unique_names(src, out, paths, dsts):
    """Two picks with one name in one folder (class folders gather images from everywhere), or a name an earlier
    pick already left there: the naming engine (core.naming) gives them the folder prefix — only those."""
    from AIO_QC.core import naming as N
    items, existing = [], {}
    for i, (p, d) in enumerate(zip(paths, dsts)):
        folder = os.path.relpath(os.path.dirname(d), out).replace("\\", "/").lower()
        items.append(N.Item(i, os.path.relpath(p, src).replace("\\", "/"), split=folder, path=p,
                            extra_exts=(".json",)))
        if folder not in existing:
            try:
                existing[folder] = {N.name_key(os.path.splitext(f)[0]) for f in os.listdir(os.path.dirname(d))}
            except OSError:
                existing[folder] = set()
    seen = {}
    for it in items:
        seen[(it.split, it.nk)] = seen.get((it.split, it.nk), 0) + 1
    touch = {it.key for it in items if seen[(it.split, it.nk)] > 1 or it.nk in existing.get(it.split, ())}
    if not touch:
        return dsts, []
    NP = N.plan(items, N.Scheme(preset="folder"), flat=True, existing=existing, root_len=len(out) + 40, touch=touch)
    if not NP.ok:
        NP = N.resolve(NP, existing=existing, root_len=len(out) + 40)
    out_d, renamed = list(dsts), []
    for it in NP.items:
        stem = NP.new[it.key]
        if stem != it.stem:
            d = os.path.join(os.path.dirname(dsts[it.key]), stem + it.suffix)
            renamed.append(f"{it.rel}  →  {os.path.relpath(d, out)}")
            out_d[it.key] = d
    return out_d, renamed


def _safe_dir(name: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in str(name)).strip(" .") or "_"


def cls_uncertainty(rows) -> tuple[float, int | None]:
    """A classifier's answer for one image (whole-frame rows: conf, class id) -> (uncertainty 0 … 1 = 1 - margin
    between the best two classes, best class id)."""
    import numpy as np
    a = np.asarray(rows, np.float32).reshape(-1, 6) if rows is not None and len(rows) else None
    if a is None:
        return 1.0, None
    c = np.sort(a[:, 4])[::-1]
    best = int(a[int(np.argmax(a[:, 4])), 5])
    return float(1.0 - (c[0] - (c[1] if len(c) > 1 else 0.0))), best


def run(P, progress=None, cancel=None) -> dict:
    o = P.opts
    os.makedirs(o["out"], exist_ok=True)
    log = C.RunLog(os.path.join(o["out"], C.LOG_DIR), "pick", keep_originals=False)
    st = {"done": 0, "errors": [], "log": str(log.csv_path), "undo": str(log.jsonl_path)}
    rows = []
    try:
        for i, (_k, src, dst, _x) in enumerate(P.actions, 1):
            if cancel is not None and cancel.is_set():
                st["errors"].append("stopped by you")
                break
            try:
                if os.path.exists(dst):
                    raise FileExistsError(f"{dst} already exists")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if o["copy"]:
                    shutil.copy2(src, dst)
                    log.created(dst, "picked", src, "copy")
                else:
                    shutil.move(src, dst)
                    log.moved(src, dst, "picked")
                nov, unc = o["scores"][i - 1] if i - 1 < len(o["scores"]) else (None, None)
                rows.append([i, os.path.relpath(dst, o["out"]), "" if nov is None or nov != nov else f"{nov:.4f}",
                             "" if unc is None else f"{unc:.4f}", src])
                st["done"] += 1
            except Exception as exc:
                log.error("pick", src, dst, exc)
                st["errors"].append(f"{src}: {exc}")
            if progress is not None and (i % 50 == 0 or i == len(P.actions)):
                progress(i, len(P.actions), "Copying" if o["copy"] else "Moving")
        lst = os.path.join(o["out"], "picked.csv")
        with open(lst, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["rank", "image", "novelty", "model uncertainty", "source"])
            w.writerows(rows)
        log.created(lst, "ranked list", "", "list")
    finally:
        log.close()
    return st
