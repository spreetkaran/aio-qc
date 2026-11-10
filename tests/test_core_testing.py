# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Testing / Frame Extractor core (Qt-free) and the motion engine's regressions."""
from __future__ import annotations

import gc
import os
import shutil
import threading

import numpy as np
import pytest

from .fx_fixtures import D

MODEL_A = os.path.join(D, "models", "dummy_a.onnx")
MODEL_B = os.path.join(D, "models", "dummy_b.onnx")


# ------------------------------------------------------------------ metrics
def test_metrics_perfect_and_empty():
    from AIO_QC.core.testing import metrics as M
    names = {0: "car", 1: "plate"}
    gts = [(np.array([[10, 10, 50, 50], [60, 60, 90, 90]], np.float32), np.array([0, 1])) for _ in range(5)]
    perfect = [np.array([[10, 10, 50, 50, 0.9, 0], [60, 60, 90, 90, 0.8, 1]], np.float32) for _ in range(5)]
    r = M.evaluate("p", names, gts, perfect)
    assert r.map50 == pytest.approx(0.995, abs=1e-3)               # 101-point interpolation, as YOLO val
    none = [np.zeros((0, 6), np.float32) for _ in range(5)]
    r0 = M.evaluate("n", names, gts, none)
    assert r0.map50 == 0.0
    mp, matched, un = M.build_class_map({0: "Car", 1: "truck"}, names)
    assert mp == {0: 0, 1: -1} and un == ["truck"]


# ------------------------------------------------------------------ runner + cache (resumable, instant re-run)
def test_runner_cache_resume(tmp_path):
    from AIO_QC.core.testing import pipeline as P
    from AIO_QC.core.testing.runner import PredCache, run_on_images
    data = P.open_data(os.path.join(D, "ts"))
    cache = PredCache(str(tmp_path / "p.db"))
    spec = P.spec_from_cfg(MODEL_A, {})
    imgs = data.images
    info = run_on_images([spec], imgs, cache)
    assert info[spec.key]["ran"] == len(imgs)
    info2 = run_on_images([spec], imgs, cache)
    assert info2[spec.key].get("ran", 0) == 0                    # everything from the cache
    spec.class_map = {"0": "plate"}                              # a class map never changes the cache key
    assert spec.key == P.spec_from_cfg(MODEL_A, {}).key


# ------------------------------------------------------------------ evaluation, report, video run
def test_evaluate_report_video_run(tmp_path):
    from AIO_QC.core.testing import pipeline as P
    data = P.open_data(os.path.join(D, "ts"))
    cache = P.cache_for(None)
    specs = P.unique_labels([P.spec_from_cfg(MODEL_A, {}), P.spec_from_cfg(MODEL_B, {})])
    run, infos = P.evaluate(data, specs, cache, only_reviewed=False)
    assert run.scored and run.ranking()
    for kind in ("summary", "detailed", "comparison"):
        paths = P.make_report(str(tmp_path / f"{kind}.html"), kind, run, [], "t", pair=(0, 1) if kind == "comparison"
                              else None, exports=kind == "detailed")
        assert os.path.getsize(paths[0]) > 10_000
    vids = P.analyse_videos([os.path.join(D, "a.mp4")], specs, cache, every_s=0.5)
    assert vids and vids[0][1]["specs"]
    vr = P.video_run(vids, specs, cache, "auto", 0, None)
    assert vr is not None and vr.ground == "reference" and len(vr.items) <= P.VIDEO_RUN_MAX_FRAMES
    paths = P.make_report(str(tmp_path / "video.html"), "video", None, vids, "v")
    assert os.path.isfile(paths[0])
    P.release_reader()


def test_video_samples_capped(monkeypatch):
    from AIO_QC.core.testing import pipeline as P
    stride, step, n = P.video_plan(os.path.join(D, "a.mp4"), 0.04, max_samples=10)
    assert n <= 10 and stride >= 1


# ------------------------------------------------------------------ test sets
def test_testset_build_dedup(tmp_path):
    from AIO_QC.core.testing.testset import TestSet
    ts = TestSet.create(str(tmp_path / "ts"))
    a = os.path.join(D, "a.mp4")
    copy = str(tmp_path / "same_scene_other_camera.mp4")
    shutil.copy(a, copy)
    st = ts.add_sources([dict(type="video", path=a, start=0, end=3), dict(type="video", path=a, start=1, end=4)],
                        dict(ts.manifest["settings"], sampling="interval", interval_s=1.0, unique="balanced"))
    n = st["added"]
    assert n == len(ts.items) and len({it["file"] for it in ts.items}) == n          # overlap merged, no clash
    st2 = ts.add_sources([dict(type="video", path=copy, start=0, end=4)],
                         dict(ts.manifest["settings"], sampling="interval", interval_s=1.0, unique="balanced"))
    assert st2["added"] == 0 and st2["duplicates"] >= 4                             # across videos
    assert os.path.isfile(os.path.join(ts.root, "data.yaml"))


# ------------------------------------------------------------------ frame search (classic embedder, no model)
def test_frame_search_classic(tmp_path, monkeypatch):
    from AIO_QC.core.testing import search as S
    monkeypatch.setattr(S, "search_dir", lambda: str(tmp_path))
    from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
    emb = S.ClassicEmbedder()
    ix = S.VideoIndex(os.path.join(D, "a.mp4"), emb.id)
    ix.build(emb, find_ffmpeg(), interval=0.5)
    assert ix.vecs is not None and len(ix.vecs) >= 8
    hits = S.search([ix], ix.vecs[3], k=3)
    assert hits and abs(hits[0]["time"] - float(ix.times[3])) < 1e-3


# ------------------------------------------------------------------ provenance of extracted frames
def test_frame_sources(tmp_path):
    from AIO_QC.tools.frame_extractor.core import frame_sources as F
    img = tmp_path / "sub" / "x_frame_12.jpg"
    img.parent.mkdir()
    img.write_bytes(b"x")
    F.record(str(img), os.path.join(D, "a.mp4"), 12, 0.48)
    r = F.lookup(str(img))
    assert r["frame"] == 12 and r["video"].endswith("a.mp4")
    loose = tmp_path / "a_frame_7.jpg"
    loose.write_bytes(b"x")
    assert F.lookup(str(loose), [os.path.join(D, "a.mp4")])["frame"] == 7
    ts_img = os.path.join(D, "ts", "images", sorted(os.listdir(os.path.join(D, "ts", "images")))[1])
    assert F.lookup(ts_img)["video"].endswith(".mp4")


# ------------------------------------------------------------------ motion: regressions
def test_tab_transition_with_temporary_wrappers(qapp, qtbot):
    """animate_tabs() hosts its cross-fade on page.parentWidget() - a TEMPORARY wrapper of a widget Qt created.
    A weak reference would think it dead once the caller dropped it and leave the snapshot stuck on screen."""
    from PyQt6.QtWidgets import QLabel, QTabWidget
    from AIO_QC.widgets import motion
    tabs = QTabWidget()
    for i in range(4):
        tabs.addTab(QLabel(f"page {i}" * 20), f"t{i}")
    motion.animate_tabs(tabs)
    qtbot.addWidget(tabs)
    tabs.resize(400, 300)
    tabs.show()
    qtbot.wait(50)
    for i in (1, 2, 3, 0, 2, 1):
        tabs.setCurrentIndex(i)
        gc.collect()                                   # drop every temporary wrapper
        qtbot.wait(20)
    qtbot.wait(900)                                    # longer than any transition + its watchdog
    stuck = [w for w in tabs.findChildren(motion.Transition) if w.isVisible()]
    assert not stuck and not motion._OVERLAYS


def test_overlay_watchdog_when_animation_cannot_run(qapp, qtbot, monkeypatch):
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QWidget
    from AIO_QC.widgets import motion
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 100)
    host.show()
    monkeypatch.setattr(motion, "animate", lambda *a, **k: None)          # the clock never runs
    pm = QPixmap(200, 100)
    motion.Transition(host, pm, 0, 0, 100)
    qtbot.waitUntil(lambda: not [w for w in host.findChildren(motion.Transition) if w.isVisible()], timeout=2000)


def test_no_animation_off_the_ui_thread(qapp):
    from AIO_QC.widgets import motion
    got, err = [], []

    def worker():
        try:
            r = motion.animate(("t", 1), got.append, 0.0, 1.0, 300)
            got.append(("ret", r))
        except Exception as e:                          # noqa: BLE001
            err.append(e)
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not err and got[0] == 1.0 and got[-1] == ("ret", None)          # end state at once, no tween
    assert ("t", 1) not in motion._TWEENS
