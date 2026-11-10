# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Prediction Explorer (Model Check ▸ 🔭 Explorer): core matching / views / curves, and the whole UI on real Qt with a
fake inference server (two models, one with segmentation outlines)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def rect(l, x1, y1, x2, y2):
    return {"label": l, "shape_type": "rectangle", "points": [[x1, y1], [x2, y2]]}


def _small():
    from AIO_QC.core import explorer as E
    b = E.Builder(["A", "B"], {"A": "mA", "B": "mB"}, {"A": "#f00", "B": "#0f0"}, {"car": {"A", "B"}, "plate": {"A"}})
    b.add("k0", "train", "", 100, 100, [rect("car", 10, 10, 50, 50), rect("plate", 60, 60, 80, 70)], False,
          {"A": [("car", 0.9, (11, 11, 50, 50))], "B": [("car", 0.8, (10, 10, 49, 51)), ("car", 0.7, (70, 5, 95, 30))]},
          {"A": True})
    b.add("k1", "val", "f", 100, 100, [rect("car", 10, 10, 50, 50), rect("plate", 60, 60, 90, 90)], False,
          {"A": [("plate", 0.6, (10, 10, 50, 50)), ("plate", 0.5, (60, 60, 75, 75), [(60, 60), (75, 60), (70, 75)])]})
    b.add("k2", "val", "f", 100, 100, [], False, {})
    return b.finish()


def tags_of(E, d, st, i):
    return [(o.kind, o.label, E.TAG_NAMES[o.tag], o.model) for o in E.objects(d, st, i)]


def test_core_matching_views_edits(tmp_path):
    from AIO_QC.core import explorer as E
    d = _small()
    v = E.View(models={"A": {"visible": True, "conf": 0.25}, "B": {"visible": True, "conf": 0.25}})
    st = E.evaluate(d, v)
    assert tags_of(E, d, st, 0) == [("gt", "car", "correct", ""), ("gt", "plate", "missed", ""),
                                    ("pred", "car", "correct", "A"), ("pred", "car", "correct", "B"),
                                    ("pred", "car", "not labelled", "B")]
    assert tags_of(E, d, st, 1) == [("gt", "car", "wrong class", ""), ("gt", "plate", "poor fit", ""),
                                    ("pred", "plate", "wrong class", "A"), ("pred", "plate", "poor fit", "A")]
    assert list(st.order) == [1, 0, 2]                                 # most errors first
    assert st.tag_counts["models disagree"] == 3
    poly = [o for o in E.objects(d, st, 1) if o.kind == "pred"][1].poly
    assert poly is not None and len(poly) == 3                         # segmentation outline kept
    # filters
    v.tags = ["missed"]
    assert list(E.evaluate(d, v).order) == [0]
    v.tags, v.classes = None, ["plate"]
    st = E.evaluate(d, v)
    assert {o.label for i in st.order for o in E.objects(d, st, i)} == {"plate"}
    v.classes = None
    v.models["B"]["visible"] = False
    st = E.evaluate(d, v)
    assert ("pred", "car", "not labelled", "B") not in tags_of(E, d, st, 0)
    # thresholds hide predictions without re-matching
    v.models["A"]["conf"] = 0.95
    st = E.evaluate(d, v)
    assert all(o.kind == "gt" for o in E.objects(d, st, 0))
    assert tags_of(E, d, st, 0)[0][2] == "missed"
    v.models["A"]["conf"] = 0.25
    st = E.evaluate(d, v)
    # suggestions -> session ops
    ops = [E.op_for(d, st, o, 0.5) for o in E.objects(d, st, 1) if o.kind == "pred"]
    assert ops[0] == ("k1", ("relabel", 0, (0.1, 0.1, 0.5, 0.5), "plate"))
    assert ops[1][1][0] == "reshape"
    bulk = E.bulk_ops(d, st, [0, 1], {"add", "relabel", "reshape"}, 0.3, 0.5)
    assert set(bulk) == {"k1"} and len(bulk["k1"]) == 2
    # an edit re-matches only that image
    E.replace_gt(d, {"k1": ([rect("plate", 10, 10, 50, 50), rect("plate", 60, 60, 90, 90)], False, 100, 100)})
    st = E.evaluate(d, v)
    assert [t[2] for t in tags_of(E, d, st, 1)] == ["correct", "poor fit", "correct", "poor fit"]
    # save / load round trip (compacts dead rows)
    p = tmp_path / "x.npz"
    E.save(d, p)
    d2 = E.load(p)
    st2 = E.evaluate(d2, v)
    assert tags_of(E, d2, st2, 1) == tags_of(E, d, st, 1)
    assert len(d2.g_img) == 4
    # curves
    c = E.curves(d2, 0, None, 0.5)
    assert c["npos"] == 4 and c["n"] == 3 and len(c["precision"]) == 3
    p_, r_, n_ = E.at_threshold(c, 0.55)
    assert n_ == 2 and 0 <= p_ <= 1 and 0 <= r_ <= 1
    # patches
    codes = E.patch_rows(d2, st2, "errors")
    kinds = {E.decode_id(int(c))[0] for c in codes}
    assert kinds == {"gt", "pred"}
    # views persist
    E.save_views(tmp_path, {"mine": v.to_dict()})
    assert E.View.from_dict(E.load_views(tmp_path)["mine"]).models["B"]["visible"] is False


def test_core_scale_and_embedding_map(tmp_path):
    from AIO_QC.core import explorer as E
    rng = np.random.default_rng(0)
    b = E.Builder(["A", "B"], coverage={"car": {"A", "B"}})
    b.CHUNK = 700                                                      # several matching blocks
    N = 3000
    for i in range(N):
        x = rng.uniform(0, 500, (3, 2))
        shapes = [rect("car", a, c, a + 40, c + 40) for a, c in x]
        pa = [("car", float(rng.uniform()), (a + 2, c + 1, a + 41, c + 40)) for a, c in x[:2]]
        pb = [("car", float(rng.uniform()), (a + 1, c + 2, a + 40, c + 43)) for a, c in x[1:]]
        b.add(f"/r/img{i}.jpg", "train", "", 640, 640, shapes, False, {"A": pa, "B": pb})
    d = b.finish()
    v = E.View(models={"A": {"conf": 0.3}, "B": {"conf": 0.3}})
    st = E.evaluate(d, v)
    assert st.n_images == N
    # per-image greedy (the reference implementation in evaluation.py) agrees with the vectorised blocks
    from AIO_QC.core.evaluation import IOU_GRID, _greedy
    from AIO_QC.core.model_check import iou_matrix
    ti = int(np.argmin(np.abs(IOU_GRID - 0.5)))
    for i in range(0, N, 97):
        gr, pr = d.gt_rows(i), d.pred_rows(i)
        for m in (0, 1):
            pm = pr[d.p_model[pr] == m]
            M = iou_matrix(d.g_box[gr].astype(float), d.p_box[pm].astype(float))
            ref = _greedy(M, np.argsort(-d.p_conf[pm], kind="stable"), len(gr))[ti]
            got = d.match[ti][pm]
            assert list(np.where(ref >= 0, gr[np.maximum(ref, 0)], -1)) == list(got)
    # embeddings cached by Visual Similarity -> map rows
    ids = [f"img{i}.jpg|rectangle:{d.g_box[g][0]:.2f}:{d.g_box[g][1]:.2f},{d.g_box[g][2]:.2f}:{d.g_box[g][3]:.2f}|1:2"
           for i in range(50) for g in d.gt_rows(i)]
    pos, rows = E.embedded_rows(d, "/r", np.array(ids), 1000)
    assert len(pos) == 150 and set(d.g_img[rows]) == set(range(50))
    xy, used = E.project(rng.normal(size=(150, 32)).astype(np.float32), "pca")
    assert xy.shape == (150, 2) and used == "pca" and xy.min() >= 0 and xy.max() <= 1


# ===================================================================================================== UI
def _fake_models(root, tmp_path):
    from AIO_QC.core.state import save_models_config
    pa, pb = tmp_path / "a.pt", tmp_path / "b.pt"
    pa.write_bytes(b"x")
    pb.write_bytes(b"y")
    cfg = {"models": [
        {"id": "ma", "name": "det", "path": str(pa), "enabled": True, "imgsz": 320, "task": "detect",
         "names": {"0": "car", "1": "plate"}, "class_map": {"0": "car", "1": "plate"}},
        {"id": "mb", "name": "seg", "path": str(pb), "enabled": True, "imgsz": 320, "task": "segment",
         "names": {"0": "car"}, "class_map": {"0": "car"}}]}
    save_models_config(root, cfg)


def _predict(images, ids, conf=0.05, batch=8, classes=None, cancel=None):
    out = {}
    for p in images:
        i = int(re.search(r"_(\d+)\.", os.path.basename(p)).group(1))
        x = 40 + i % 50
        r = {"w": 320, "h": 240, "by_model": {}, "geo": {}}
        if "ma" in ids:
            boxes = [[x + 1, 41, x + 99, 119, 0.9, 0]]
            if i % 3 == 0:
                boxes.append([250, 10, 300, 60, 0.8, 0])             # an object nobody labelled
            if i % 4 == 0:
                boxes.append([x + 10, 85, x + 62, 105, 0.7, 0])      # the plate called a car
            r["by_model"]["ma"] = boxes
        if "mb" in ids:
            r["by_model"]["mb"] = [[x + 30, 55, x + 80, 105, 0.6, 0]]        # loose: poor fit
            r["geo"]["mb"] = [{"poly": [[x + 20, 50], [x + 90, 50], [x + 90, 110], [x + 20, 110], [x + 55, 115]]}]
        out[p] = r
    return out


@pytest.fixture
def mc(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    _fake_models(w.session.root, tmp_path)
    svc = w.inference
    monkeypatch.setattr(svc, "python", "/fake/python")
    monkeypatch.setattr(svc, "ensure_loaded", lambda specs, *a, **k: None)
    monkeypatch.setattr(svc, "predict", _predict)
    w.open_tool("model_check")
    d = w.open_dialogs["model_check"]
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    qtbot.waitUntil(lambda: d.explorer.state is not None, timeout=20000)
    return w, d


def test_explorer_ui(mc, qtbot, monkeypatch):
    from AIO_QC.core import explorer as E
    w, dlg = mc
    p = dlg.explorer
    d = p.data
    assert len(d) == 12 and set(d.models) == {"ma", "mb"}
    assert dlg.tabs.currentWidget() is p
    assert p.grid.model_.rowCount() == 12
    assert p.state.tag_counts["not labelled"] >= 4 and p.state.tag_counts["poor fit"] >= 1
    assert any(d.p_poly(r) is not None for r in range(len(d.p_img)))       # segmentation outlines kept
    assert "plate" in d.labels and "wheel" in d.labels
    # tiles paint (images decode in the pool)
    p.grid.resize(900, 700)
    p.grid.show()
    qtbot.waitUntil(lambda: len(p.grid.cache.cache) > 0, timeout=10000)
    p.grid.grab()
    # threshold: hide model "det" below 0.85
    row = p.sidebar.model_row("ma")
    row.set_conf(0.85, emit=True)
    qtbot.waitUntil(lambda: p.view.conf_of("ma") == 0.85, timeout=3000)
    assert int((p.state.p_show & (d.p_model == 0)).sum()) == 12
    row.set_conf(0.25, emit=True)
    qtbot.waitUntil(lambda: p.view.conf_of("ma") == 0.25, timeout=3000)
    # result filter: only "not labelled"
    p.sidebar.tags.set_all(True, only=["not labelled"])
    qtbot.waitUntil(lambda: p.view.tags == ["not labelled"], timeout=3000)
    assert p.state.n_images == 4
    objs = p.tile_objects(int(p.state.order[0]))
    assert objs and all(o.tag == E.EXTRA for o in objs)
    p.sidebar.tags.set_all(True)
    qtbot.waitUntil(lambda: p.view.tags is None, timeout=3000)
    p.sort.setCurrentIndex(p.sort.findData("extra"))
    # sort + search + colour by model
    p.search.setText("0003")
    qtbot.waitUntil(lambda: p.state.n_images == 1, timeout=3000)
    p.search.setText("")
    p.sidebar.color.setCurrentIndex(p.sidebar.color.findData("model"))
    qtbot.waitUntil(lambda: p.view.color == "model", timeout=3000)
    # viewer: open, accept the unlabelled box (adds a label), undo
    first = int(p.state.order[0])
    key = d.keys[first]
    n_before = len(json.load(open(key))["shapes"]) if key.endswith(".json") else None
    p.open_image(first)
    v = p.viewer
    qtbot.waitUntil(lambda: v._img is not None, timeout=10000)
    extra = next(r for r, o in enumerate(v.objs) if o.tag == E.EXTRA)
    v.table.selectRow(extra)
    assert v.accept_btn.isEnabled()
    v.accept_selected()
    qtbot.waitUntil(lambda: p.undo_btn.isEnabled(), timeout=3000)
    ann_path = w.session.by_key[key].ann_path
    shapes = json.load(open(ann_path))["shapes"]
    assert len(shapes) == (n_before or 3) + 1 and shapes[-1]["label"] == "car"
    qtbot.waitUntil(lambda: all(o.tag != E.EXTRA for o in E.objects(p.data, p.state, first)), timeout=5000)
    p.undo()
    assert len(json.load(open(ann_path))["shapes"]) == (n_before or 3)
    qtbot.waitUntil(lambda: any(o.tag == E.EXTRA for o in E.objects(p.data, p.state, first)), timeout=5000)
    # compare two models side by side, step through images
    v.compare.setChecked(True)
    assert v.canvas_b.isVisible() and v.canvas.objs and v.canvas_b.objs
    assert {o.model for o in v.canvas_b.objs if o.kind == "pred"} <= {"mb"}
    v.step(1)
    v.step(-1)
    v.grab()
    v.close()
    # objects view, relabel a label
    p.set_mode("objects")
    p.psource.setCurrentIndex(p.psource.findData("gt"))
    assert p.grid.model_.rowCount() == len(d.gt_rows(0)) * 0 + int(p.state.g_show.sum())
    p.grid.selectionModel().select(p.grid.model_.index(0), p.grid.selectionModel().SelectionFlag.Select)
    target = p.selected_objects()[0]
    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getItem", staticmethod(lambda *a, **k: ("plate", True)))
    p.relabel_selection()
    qtbot.waitUntil(lambda: p.undo_btn.isEnabled(), timeout=3000)
    tk = d.keys[int(d.g_img[target.row])]
    assert json.load(open(w.session.by_key[tk].ann_path))["shapes"][target.shape_index]["label"] == "plate"
    p.undo()
    p.grid.grab()
    # thresholds panel drives the sidebar
    p.thr_btn.setChecked(True)
    p.ins_model.setCurrentIndex(0)
    p._hist_threshold(0.5)
    qtbot.waitUntil(lambda: p.view.conf_of(p.ins_model.currentData()) == 0.5, timeout=3000)
    assert "precision" in p.ins_text.text()
    p.insight.grab()
    # saved views
    monkeypatch.setattr("PyQt6.QtWidgets.QInputDialog.getText", staticmethod(lambda *a, **k: ("strict", True)))
    p._save_view()
    assert "strict" in E.load_views(w.session.root)
    p.reset_view()
    p.views.setCurrentIndex(p.views.findData("strict"))
    p._apply_saved_view()
    qtbot.waitUntil(lambda: p.view.conf_of(p.ins_model.currentData()) == 0.5, timeout=3000)
    # analysis drill-down lands in the explorer
    p.set_mode("grid")
    dlg._to_explorer([d.keys[0], d.keys[1]], "test drill-down")
    qtbot.waitUntil(lambda: p.state.n_images == 2, timeout=3000)
    assert p.banner.isVisible()
    p.clear_keys()
    qtbot.waitUntil(lambda: p.state.n_images == 12, timeout=3000)
    # predictions layer in the main viewer
    w.set_pred_layer(True)
    w.open_sample(d.keys[0])
    qtbot.waitUntil(lambda: w.current_key == d.keys[0], timeout=5000)
    w.refresh_pred_layer()
    assert w.canvas.pred_items
    w.set_pred_layer(False)
    assert not w.canvas.pred_items


def test_explorer_map_and_reopen(mc, qtbot):
    from AIO_QC.core import explorer as E
    from AIO_QC.core.embeddings import geom_id
    w, dlg = mc
    p = dlg.explorer
    root = Path(w.session.root)
    # a Visual Similarity cache for the car labels
    ids = []
    for s in w.session.samples:
        ann = w.session.adapter.load(s, light=True)
        rel = os.path.relpath(s.key, root).replace("\\", "/")
        for sh in ann.shapes:
            if sh["label"] == "car":
                ids.append(f"{rel}|{geom_id(sh)}|1:1")
    (root / ".aio_qc" / "embeddings").mkdir(parents=True, exist_ok=True)
    np.savez(root / ".aio_qc" / "embeddings" / "abc_shape.npz", ids=np.array(ids),
             vecs=np.random.default_rng(1).normal(size=(len(ids), 16)).astype(np.float16))
    p.set_mode("map")
    assert p.map_src.currentData() is not None
    p.compute_map()
    qtbot.waitUntil(lambda: p._map is not None, timeout=20000)
    assert len(p._map[0]) == len(ids)
    p.map.resize(500, 400)
    p.map.grab()
    p._map_selected(np.arange(3))
    p._map_to("objects")
    assert p.grid.model_.rowCount() == 3
    # close (saves) and reopen: the last run comes back without inference
    dlg.close()
    qtbot.wait(50)
    assert os.path.isfile(E.data_path(root))
    # a file edited while Model Check was closed: its labels are re-read on reopen
    k0 = sorted(w.session.by_key)[0]
    ap = w.session.by_key[k0].ann_path
    js = json.load(open(ap))
    js["shapes"] = js["shapes"][:1]
    json.dump(js, open(ap, "w"))
    t = os.path.getmtime(E.data_path(root)) + 5
    os.utime(ap, (t, t))
    w.open_tool("model_check")
    d2 = w.open_dialogs["model_check"]
    qtbot.waitUntil(lambda: d2.explorer.state is not None, timeout=20000)
    ex = d2.explorer
    assert len(ex.data) == 12 and "last run" in ex.status.text()
    assert len(ex.data.gt_rows(ex.data.index_of(k0))) == 1
    d2.close()


def test_server_geometry():
    """infer_server keeps segmentation outlines / keypoints aligned with the (class-filtered) boxes."""
    from types import SimpleNamespace as NS
    from AIO_QC.core import infer_server as S

    class T:
        def __init__(self, a):
            self.a = np.asarray(a, float)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

        def __len__(self):
            return len(self.a)
    boxes = NS(xyxy=T([[0, 0, 10, 10], [5, 5, 20, 20]]), conf=T([0.9, 0.4]), cls=T([0, 1]))
    boxes.__len__ = lambda: 2
    r = NS(boxes=type("B", (), {"__len__": lambda s: 2, "xyxy": boxes.xyxy, "conf": boxes.conf, "cls": boxes.cls})(),
           masks=NS(xy=[np.array([[0, 0], [10, 0], [10, 10], [0, 10]] * 30, float), np.array([[5, 5], [20, 5], [9, 20]])]),
           obb=None, keypoints=None)
    res = {"by_model": {}}
    S._store(res, "m", r, {1})
    assert res["by_model"]["m"] == [[5.0, 5.0, 20.0, 20.0, 0.4, 1]]
    assert res["geo"]["m"] == [{"poly": [[5.0, 5.0], [20.0, 5.0], [9.0, 20.0]]}]
    res = {"by_model": {}}
    S._store(res, "m", r, None)
    assert len(res["geo"]["m"][0]["poly"]) == S.MAX_POLY                    # thinned outline
    r.masks = None
    res = {"by_model": {}}
    S._store(res, "m", r, None)
    assert "geo" not in res                                                 # plain detection: nothing extra
    from AIO_QC.core.model_check import model_signature
    assert model_signature({"path": "x.pt", "task": "segment"}).endswith("|geo1")
    assert not model_signature({"path": "x.pt", "task": "detect"}).endswith("|geo1")


def test_explorer_yolo_normalized(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    """YOLO (normalised) labels land in pixels next to the predictions; accepting adds a YOLO line."""
    from AIO_QC.core import explorer as E
    w = _open(shell, qtbot, FX.yolo(tmp_path / "y"))
    _fake_models(w.session.root, tmp_path)
    monkeypatch.setattr(w.inference, "python", "/fake/python")
    monkeypatch.setattr(w.inference, "ensure_loaded", lambda specs, *a, **k: None)
    monkeypatch.setattr(w.inference, "predict", _predict)
    w.open_tool("model_check")
    d = w.open_dialogs["model_check"]
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
    qtbot.waitUntil(lambda: d.explorer.state is not None, timeout=20000)
    p = d.explorer
    data = p.data
    assert len(data) == 16 and set(data.splits) == {"train", "val"}
    g = data.gt_rows(0)
    assert data.g_box[g].max() > 2                                   # pixels, not 0..1
    i = next(i for i in range(len(data)) if any(o.tag == E.EXTRA for o in E.objects(data, p.state, i)))
    o = next(o for o in E.objects(data, p.state, i) if o.tag == E.EXTRA)
    key = data.keys[i]
    lab = w.session.by_key[key].ann_path
    n0 = len(open(lab).read().strip().split("\n"))
    k, op = E.op_for(data, p.state, o, 0.5)
    p.apply_ops({k: [op]}, "test")
    assert len(open(lab).read().strip().split("\n")) == n0 + 1          # one more YOLO line
    qtbot.waitUntil(lambda: all(x.tag != E.EXTRA for x in E.objects(p.data, p.state, i)), timeout=5000)
    p.sidebar.splits.set_all(True, only=["val"])
    qtbot.waitUntil(lambda: p.state.n_images == 8, timeout=3000)
    d.close()
