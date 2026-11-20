# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A real classification dataset for the Class Gallery tests: scikit-learn's handwritten digits (bundled, no download)
as class folders (train / val), with planted mislabels (3 %) and cross-class copies; a cross-validated classifier's
probabilities (what Model Check would cache) and pixel-PCA vectors (what Visual Similarity would give)."""
import random
from pathlib import Path

import numpy as np


def build(out: Path, scale: int = 1) -> dict:
    from PIL import Image
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    out = Path(out)
    d = load_digits()
    X, y = d.images, d.target
    NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    rng = random.Random(7)
    items = []                                       # (path, true class, filed class, pixels)
    for rep in range(scale):
        for i in range(len(y)):
            px = X[i] / 16.0
            if rep:                       # more copies: shifted / noisy versions (still the same digit)
                px = np.clip(np.roll(px, rng.choice([-1, 0, 1]), axis=rng.choice([0, 1])) +
                             np.random.default_rng(rep * 10_000 + i).normal(0, 0.08, px.shape), 0, 1)
            split = "val" if (i + rep) % 5 == 0 else "train"
            items.append([f"{split}", int(y[i]), int(y[i]), px, f"d{rep:02d}_{i:04d}"])
    # planted mislabels: 3 % filed under a wrong class
    wrong = rng.sample(range(len(items)), int(0.03 * len(items)))
    for j in wrong:
        t = items[j][1]
        items[j][2] = rng.choice([c for c in range(10) if c != t])
    # planted cross-class duplicates: 12 images copied into another class folder
    dups = []
    for j in rng.sample([k for k in range(len(items)) if k not in set(wrong)], 12):
        sp, t, f, px, name = items[j]
        o = rng.choice([c for c in range(10) if c != t])
        dups.append(len(items))
        items.append([sp, t, o, px.copy(), name + "_copy"])
    truth = {}
    for sp, t, f, px, name in items:
        p = out / sp / NAMES[f] / f"{name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray((255 - px * 255).astype(np.uint8)).resize((96, 96), Image.NEAREST).convert("RGB")
        img.save(p)
        truth[str(p)] = {"true": NAMES[t], "filed": NAMES[f]}
    # a classifier that never saw a sample's own (filed) label: 5-fold cross-validated probabilities
    flat = np.stack([it[3].ravel() for it in items])
    filed = np.array([it[2] for it in items])
    clf = LogisticRegression(max_iter=2000, C=0.5)
    proba = cross_val_predict(clf, flat, filed, cv=5, method="predict_proba")
    vecs = PCA(n_components=24, random_state=0).fit_transform(flat).astype(np.float32)
    paths = [str(out / it[0] / NAMES[it[2]] / f"{it[4]}.png") for it in items]
    return {"paths": paths, "proba": proba.round(5).tolist(), "names": NAMES, "truth": truth,
            "wrong": [paths[j] for j in wrong], "dups": [paths[j] for j in dups], "vecs": vecs}


def onnx_classifier(out: Path, images, labels, names, c=1.0, size=8) -> Path:
    """A real (tiny) ONNX image classifier for the digits: a logistic regression on the size × size grey image the
    app's classification transform produces (shorter side → size, centre crop, RGB / 255), exported with the
    metadata an exported classifier carries (task, names, imgsz). images: BGR arrays; labels: class names."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    from sklearn.linear_model import LogisticRegression
    from AIO_QC.tools.frame_extractor.core.detector import _cls_blob
    x = _cls_blob(list(images), size).mean(1).reshape(len(images), -1)
    idx = {n: i for i, n in enumerate(names)}
    clf = LogisticRegression(C=c, max_iter=2000).fit(x, [idx[n] for n in labels])
    w = np.zeros((size * size, len(names)), np.float32)
    b = np.full(len(names), -30.0, np.float32)                      # a class it never saw: never chosen
    for k, ci in enumerate(clf.classes_):
        w[:, ci] = clf.coef_[k]
        b[ci] = clf.intercept_[k]
    nodes = [helper.make_node("ReduceMean", ["images"], ["grey"], axes=[1], keepdims=0),
             helper.make_node("Reshape", ["grey", "shape"], ["flat"]),
             helper.make_node("MatMul", ["flat", "W"], ["logits0"]),
             helper.make_node("Add", ["logits0", "B"], ["logits"]),
             helper.make_node("Softmax", ["logits"], ["output0"], axis=1)]
    inits = [helper.make_tensor("shape", TensorProto.INT64, [2], [-1, size * size]),
             helper.make_tensor("W", TensorProto.FLOAT, list(w.shape), w.flatten().tolist()),
             helper.make_tensor("B", TensorProto.FLOAT, [len(names)], b.tolist())]
    g = helper.make_graph(nodes, "digits-cls",
                          [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, size, size])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", len(names)])],
                          inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    for k, v in (("task", "classify"), ("names", str({i: n for i, n in enumerate(names)})),
                 ("imgsz", str([size, size]))):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(m)
    out = Path(out)
    onnx.save(m, str(out))
    return out
