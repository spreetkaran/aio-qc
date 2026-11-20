# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Visual Similarity end to end without a real vision model: a stand-in Python runs the REAL embed_server.py, with
only the model swapped for a tiny colour embedding (so red objects look alike, blue ones look alike). Everything else
is real — the protocol, crops, the embedding cache, the nearest-neighbour search, the analysis, relabel / undo and the
hand-off to the Crop Gallery."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from .test_aio_image import _open, shell  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Python is a shell script")

TOY = r'''
import importlib.util, sys
import numpy as np
path = sys.argv[-1]
spec = importlib.util.spec_from_file_location("embed_server", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

def probe():
    return {"python": sys.version.split()[0], "torch": "toy", "transformers": "toy", "timm": None,
            "open_clip": None, "huggingface_hub": None, "faiss": None, "PIL": "x",
            "devices": [{"id": "cpu", "name": "CPU"}]}

def load(path, device="auto", half=True):
    S.STATE.update(model="toy", kind="toy", device="cpu", half=False, dim=12, path=path, size=32)
    return {"kind": "toy", "root": path, "device": "cpu", "half": False, "dim": 12, "size": 32}

def _forward(arrs):
    out = []
    for a in arrs:
        a = np.asarray(a, np.float32)
        h, w = a.shape[:2]
        v = np.concatenate([a[y:y + h // 2 or 1, x:x + w // 2 or 1].reshape(-1, 3).mean(0)
                            for y in (0, h // 2) for x in (0, w // 2)])
        out.append(v / (np.linalg.norm(v) + 1e-6))
    return np.stack(out)

S.probe, S.load, S._forward = probe, load, _forward
S.main()
'''


def _fake_python(tmp: Path) -> str:
    d = tmp / "toyenv"
    d.mkdir()
    (d / "toy_embed.py").write_text(TOY)
    py = d / "python"
    py.write_text(f'#!/bin/sh\ncase "$*" in *embed_server.py*) exec "{sys.executable}" -u "{d / "toy_embed.py"}" '
                  f'"$@" ;; esac\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    return str(py)


def _dataset(root: Path) -> Path:
    """10 images: a red and a blue square each — one blue square is labelled "red" (the mislabel)."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(10):
        img = np.full((120, 160, 3), 200, np.uint8)
        img += rng.integers(0, 8, img.shape, dtype=np.uint8)
        cv2.rectangle(img, (10, 20), (60, 70), (30, 30, 220), -1)        # BGR: red
        cv2.rectangle(img, (90, 30), (140, 80), (220, 40, 30), -1)       # blue
        cv2.imwrite(str(root / f"im{i}.jpg"), img)
        blue = "red" if i == 4 else "blue"
        shapes = [{"label": "red", "shape_type": "rectangle", "points": [[10, 20], [60, 70]], "flags": {},
                   "group_id": None, "description": ""},
                  {"label": blue, "shape_type": "rectangle", "points": [[90, 30], [140, 80]], "flags": {},
                   "group_id": None, "description": ""}]
        (root / f"im{i}.json").write_text(json.dumps({"version": "5.4.1", "flags": {}, "shapes": shapes,
                                                      "imagePath": f"im{i}.jpg", "imageData": None,
                                                      "imageHeight": 120, "imageWidth": 160}))
    return root


def _labels(root, name):
    return [s["label"] for s in json.loads((root / name).read_text())["shapes"]]


def test_visual_similarity_with_a_toy_model(shell, qtbot, tmp_path):  # noqa: F811
    root = _dataset(tmp_path / "ds")
    model = tmp_path / "toy-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "toy"}')
    w = _open(shell, qtbot, root)
    w.inference.set_python(_fake_python(tmp_path))
    w.open_tool("similarity")
    d = w.open_dialogs["similarity"]
    # no model yet: a clear message, nothing runs
    d.models = []
    d._fill_models()
    with pytest.raises(ValueError, match="Add a model"):
        d.make_context()
    d._register(model, "Toy")
    assert d.model_box.currentData() == str(model) and "Toy" in d.model_box.currentText()
    d.k.setValue(4)
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    assert d.ctx.get("vecs") is not None, d.status.text()
    assert d.ctx["vecs"].shape == (20, 12) and d.ctx["n_new"] == 20
    assert "20 objects compared with Toy" in d.summarize() and "20 newly embedded" in d.summarize()
    mis = [r for r in d.rows if r.payload and r.payload[0].kind == "mislabel"]
    assert any(r.key.endswith("im4.json") and r.payload[0].suggest == "blue" for r in mis), \
        [(Path(r.key).name, r.payload[0].kind, r.payload[0].label, r.payload[0].suggest) for r in d.rows]
    assert w.embed_index["model"] == "Toy" and len(w.embed_index["map"]) == 20
    # the second run takes every vector from the cache
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    assert d.ctx["n_new"] == 0 and "0 newly embedded, 20 from the cache" in d.summarize()
    # use the suggestion (ticked row), then undo it
    row = next(r for r in d.rows if r.payload and r.payload[0].kind == "mislabel" and r.key.endswith("im4.json"))
    from PyQt6.QtCore import Qt
    for t in range(d.table.rowCount()):
        d.table.item(t, 0).setCheckState(Qt.CheckState.Checked if d._row_at(t) is row else Qt.CheckState.Unchecked)
    assert d.target_rows() == [row]
    d._relabel()
    assert _labels(root, "im4.json") == ["red", "blue"] and d.undo_btn.isEnabled()
    d._undo()
    assert _labels(root, "im4.json") == ["red", "red"] and not d.undo_btn.isEnabled()
    # after Undo a new run finds it again (all from the cache); the Crop Gallery opens on its label
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    assert d.ctx["n_new"] == 0 and any(r.payload and r.payload[0].kind == "mislabel" for r in d.rows)
    t0 = next(t for t in range(d.table.rowCount()) if d._row_at(t) is not None and d._row_at(t).payload)
    d.table.item(t0, 0).setCheckState(Qt.CheckState.Checked)
    want = d._row_at(t0).payload[0].label
    d._gallery()
    g = w.open_dialogs["crop_gallery"]
    assert g.label == want and g.sort_box.currentText() == "Least typical first"
    g.close()
    d.close()
