# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Search in Words end to end with a stand-in model: the REAL embed_server.py with a colour embedding whose text side
knows "red" / "blue". Search objects and images (only images without a label), the instant second search (no rescan),
per-split coverage, class descriptions (saved with the dataset) finding the blue square labelled red, relabel / undo,
the per-class agreement, and an image-only model refused with a clear message. Plus the numpy core on its own."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from .test_aio_image import _open, shell  # noqa: F401
from .test_similarity_fake import TOY, _dataset, _labels

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Python is a shell script")

TEXT = TOY.replace("S.probe, S.load, S._forward = probe, load, _forward\nS.main()", r'''
def load2(path, device="auto", half=True):
    r = load(path, device, half)
    r["text"] = "notext" not in path
    S.STATE["calls"] = S.STATE.get("calls", 0)
    return r

def embed_text(texts, out):
    vs = []
    for t in texts:
        t = t.lower()
        rgb = np.array([1.0 if "red" in t else 0.0, 0.0, 1.0 if "blue" in t else 0.0]) + 0.05
        v = np.tile(rgb, 4)
        vs.append(v / np.linalg.norm(v))
    np.save(out, np.stack(vs).astype(np.float32))
    S.STATE["text_calls"] = S.STATE.get("text_calls", 0) + 1
    return {"n": len(vs), "dim": 12}

S.probe, S.load, S._forward, S.embed_text = probe, load2, _forward, embed_text
S.main()''')


def _py(tmp: Path) -> str:
    d = tmp / "toyenv"
    d.mkdir()
    (d / "toy_embed.py").write_text(TEXT)
    py = d / "python"
    py.write_text(f'#!/bin/sh\ncase "$*" in *embed_server.py*) exec "{sys.executable}" -u "{d / "toy_embed.py"}" '
                  f'"$@" ;; esac\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    return str(py)


def _run(d, qtbot):
    d.status.setText("")
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None and d.status.text().startswith(("Done", "⚠")), timeout=60000)


def test_search_in_words_and_class_descriptions(shell, qtbot, tmp_path):  # noqa: F811
    root = _dataset(tmp_path / "ds")
    model = tmp_path / "toy-clip"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "toy"}')
    w = _open(shell, qtbot, root)
    w.inference.set_python(_py(tmp_path))
    w.open_tool("words")
    d = w.open_dialogs["words"]
    d._register(model, "ToyCLIP")
    # objects: "blue" → the 10 blue squares first (one of them labelled "red")
    d.mode.setCurrentIndex(d.mode.findData("search"))
    d.what.setCurrentIndex(d.what.findData("shape"))
    d.query.setText("blue")
    d.top.setValue(10)
    _run(d, qtbot)
    assert len(d.rows) == 10, d.status.text()
    hits = [(Path(r.key).name, r.shape_index, r.cells[2]) for r in d.rows]
    assert all(si == 1 for _n, si, _l in hits) and ("im4.json", 1, "red") in hits
    assert d.ctx["n_new"] == 20 and "best 10 of 20 objects" in d.summarize()
    # a new query is instant: vectors from memory, only the words embedded
    d._qw = None
    d.query.setText("red")
    _run(d, qtbot)
    assert all(r.shape_index == 0 for r in d.rows) and d.ctx["n_new"] == 0 and d._qw is not None   # no rescan
    # images: only images WITHOUT a 'blue' label → im4 (its blue square is labelled red)
    d.what.setCurrentIndex(d.what.findData("image"))
    d.without.setCurrentIndex(d.without.findData("blue"))
    d.query.setText("blue square")
    _run(d, qtbot)
    assert [Path(r.key).name for r in d.rows] == ["im4.json"] and "without 'blue'" in d.summarize()
    # class descriptions: saved with the dataset; the check finds the blue square labelled red
    from AIO_QC.core import text_search as TS
    TS.save_descriptions(root, {"template": "a photo of {}.", "classes": {"red": "a red square",
                                                                           "blue": "a blue square"}})
    d.mode.setCurrentIndex(d.mode.findData("desc"))
    d.without.setCurrentIndex(0)
    _run(d, qtbot)
    other = [r for r in d.rows if r.payload and r.payload[0].kind == "other"]
    assert [(Path(r.key).name, r.payload[0].label, r.payload[0].suggest) for r in other] == [("im4.json", "red",
                                                                                              "blue")]
    assert "agrees with the labels on 95%" in d.summarize(), d.summarize()
    rep = d.ctx["report"]
    assert rep.agreement["blue"] == 1.0 and rep.confused["red"][0] == "blue"
    got = []
    d.report = lambda title, lines, errs: got.append(lines)
    d._agreement()
    assert any("red: 91%" in ln for ln in got[0]), got
    # relabel the ticked row, then undo
    from PyQt6.QtCore import Qt
    for t in range(d.table.rowCount()):
        d.table.item(t, 0).setCheckState(Qt.CheckState.Checked if d._row_at(t) is other[0] else
                                         Qt.CheckState.Unchecked)
    d._relabel()
    assert _labels(root, "im4.json") == ["red", "blue"]
    d._undo()
    assert _labels(root, "im4.json") == ["red", "red"]
    # an image-only model: refused with a clear message, nothing crashes
    m2 = tmp_path / "notext-dino"
    m2.mkdir()
    (m2 / "config.json").write_text('{"model_type": "toy"}')
    d._register(m2, "Dino")
    d.mode.setCurrentIndex(d.mode.findData("search"))
    d.query.setText("red")
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    assert "understands images only" in d.status.text() + " ".join(d.errors) + d.scan_error + d.summary.text(), \
        (d.status.text(), d.errors)
    d.close()


def test_core_rank_and_descriptions(tmp_path):
    from AIO_QC.core import text_search as TS
    rng = np.random.default_rng(0)
    a, b = np.eye(8)[0], np.eye(8)[1]
    vecs = np.stack([a + rng.normal(0, .1, 8) for _ in range(30)] + [b + rng.normal(0, .1, 8) for _ in range(30)])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    labels = ["cat"] * 30 + ["dog"] * 30
    labels[3] = "dog"                                                    # a cat labelled dog
    order, sc = TS.rank(vecs, a, top=5)
    assert all(i < 30 for i in order) and list(sc) == sorted(sc, reverse=True)
    rep = TS.check_descriptions(vecs, labels, ["cat", "dog"], np.stack([a, b]))
    assert [(f.i, f.kind, f.suggest) for f in rep.findings if f.kind == "other"] == [(3, "other", "cat")]
    assert TS.split_counts(["train", "val"], ["train"] * 3 + ["val"] * 2 + ["test"]) == \
        [("train", 1, 3), ("val", 1, 2), ("test", 0, 1)]
    assert TS.load_descriptions(tmp_path)["classes"] == {}
    TS.save_descriptions(tmp_path, {"template": "", "classes": {"x_y": "a thing"}})
    d = TS.load_descriptions(tmp_path)
    assert TS.descriptions_for(["x_y", "a_b"], d) == ["a thing", "a b"] and TS.phrase("a b", d["template"]) == "a b"
    assert json.loads((tmp_path / ".aio_qc" / TS.DESC_FILE).read_text())["classes"] == {"x_y": "a thing"}
