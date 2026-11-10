# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Video review, end to end: Testing ▸ Review videos builds the playlist and only hands it over; the Frame Extractor
plays the videos one after another with the detections of the frame on screen, then returns to the video that was
open before."""
from __future__ import annotations

import os

import pytest

from .fx_fixtures import D
from .test_modes import shell3  # noqa: F401  (fixture)

MODELS = [os.path.join(D, "models", m) for m in ("dummy_a.onnx", "dummy_b.onnx")]


@pytest.fixture(autouse=True)
def software_canvas():
    """Offscreen Qt never presents GL frames (no swaps), so playback would not advance: the software canvas plays.
    The GPU canvas + libmpv path is covered by the real-OpenGL run (test_libmpv, AIOQC_TEST_LIBMPV)."""
    from AIO_QC.utils.fx_settings import fx_settings
    st = fx_settings()
    old = st.value("ui/renderer", "auto")
    st.setValue("ui/renderer", "software")
    yield
    st.setValue("ui/renderer", old)


def _testing(sh, qtbot):
    img = sh.window("image")
    qtbot.addWidget(img)
    lab = img.open_testing()
    qtbot.addWidget(lab)
    qtbot.wait(400)                                        # the restored session is applied first
    qtbot.waitUntil(lambda: lab.job is None, timeout=60000)
    lab._add_models([dict(path=p, label=os.path.splitext(os.path.basename(p))[0], imgsz="320") for p in MODELS])
    return lab


def test_playlist_round_trip_and_validation(tmp_path):
    from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
    from AIO_QC.core.testing.runner import ModelSpec
    pl = ReviewPlaylist(items=[ReviewItem(os.path.join(D, "a.mp4"), 1.0, 5.5)],
                        specs=[ModelSpec(MODELS[0], imgsz=320)], colors=["#123456"], mode="fixed", stride=3)
    assert pl.validate() == []
    f = tmp_path / "p.json"
    pl.save(f)
    back = ReviewPlaylist.load(f)
    assert back.to_dict() == pl.to_dict() and back.specs[0].key == pl.specs[0].key
    bad = ReviewPlaylist(items=[ReviewItem("/nope.mp4", 5, 2)], specs=[], mode="zz")
    p = bad.validate()
    assert any("no models" in x for x in p) and any("missing video" in x for x in p)
    assert any("out point" in x for x in p) and any("mode" in x for x in p)


def test_model_details_card():
    from AIO_QC.core.testing.review import describe, describe_text
    from AIO_QC.core.testing.runner import ModelSpec
    info = describe(ModelSpec(MODELS[0]))
    assert info["error"] is None and info["format"] == "ONNX" and info["imgsz"] == 320
    assert info["nc"] == 2 and info["names"] == {0: "car", 1: "plate"} and info["dynamic_batch"] is True
    html = describe_text(info)
    assert "car, plate" in html and "input 320" in html
    pt = describe(type("S", (), dict(path="/tmp/x.pt", label="x", imgsz=0, key="k", load=lambda self: 1))())
    assert pt["error"] == "file not found"


def test_testing_page_builds_the_playlist(shell3, qtbot, tmp_path):  # noqa: F811
    lab = _testing(shell3, qtbot)
    lab.nav.setCurrentRow(5)
    assert lab.pages.currentIndex() == lab.REVIEW
    assert not lab.rv_go.isEnabled()                       # no videos yet
    for v in ("a.mp4", "b.mp4"):
        lab._review_add(os.path.join(D, v))
    lab.rv_table.item(0, 2).setText("0:00:04")             # out point of the first video
    qtbot.waitUntil(lambda: all(lab._rv_info.get(p) for p in MODELS), timeout=30000)
    lab._show_review_models()
    assert lab.rv_go.isEnabled(), lab.rv_hint.text()
    assert "car, plate" in lab.rv_models.text()
    vids = lab._review_videos()
    assert [os.path.basename(v["path"]) for v in vids] == ["a.mp4", "b.mp4"] and vids[0]["end"] == 4.0
    st = lab._review_state()
    lab._apply_review_state({})
    assert lab.rv_table.rowCount() == 0
    lab._apply_review_state(st)
    assert lab._review_videos() == vids
    sent = []
    shell3.review_in_player = lambda pl: sent.append(pl)
    lab._review_go()
    pl = sent[0]
    assert [i.name for i in pl.items] == ["a.mp4", "b.mp4"] and len(pl.specs) == 2 and len(pl.colors) == 2
    assert pl.items[0].end == 4.0 and pl.validate() == []
    lab.close()


def test_player_plays_the_playlist_and_returns(shell3, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
    from AIO_QC.core.testing.runner import ModelSpec
    sh = shell3
    fx = sh.window("video")
    qtbot.addWidget(fx)
    fx.output_dir = str(tmp_path)
    fx.load_video(os.path.join(D, "c.avi"))                 # the video open before the review
    qtbot.waitUntil(lambda: fx.engine is not None and fx.current is not None, timeout=30000)
    fx.goto_frame(10)
    qtbot.waitUntil(lambda: fx.current_frame == 10, timeout=10000)
    pl = ReviewPlaylist(items=[ReviewItem(os.path.join(D, "a.mp4"), 2.0, 3.0), ReviewItem(os.path.join(D, "b.mp4"))],
                        specs=[ModelSpec(p, imgsz=320) for p in MODELS], colors=["#e53935", "#1e88e5"],
                        mode="balanced", conf=0.05, cache_path=str(tmp_path / "pred.db"))
    rv = sh.review_in_player(pl)
    assert sh.mode == "video" and rv is fx.review and rv.active
    qtbot.waitUntil(lambda: fx.video_path.endswith("a.mp4") and fx.is_playing, timeout=30000)
    assert fx.current_frame >= 49                           # started at the in point (2 s)
    qtbot.waitUntil(lambda: bool(fx.canvas._dets), timeout=10000)
    labels = [b[4] for b in fx.canvas._dets]
    assert any(lbl.startswith("dummy_a:") for lbl in labels) and any(lbl.startswith("dummy_b:") for lbl in labels)
    # the out point (3 s) ends the first video; the second one plays next
    qtbot.waitUntil(lambda: fx.video_path.endswith("b.mp4"), timeout=30000)
    qtbot.waitUntil(lambda: fx.is_playing, timeout=30000)
    assert "done" in rv.list.item(0).text()
    # a paused frame gets its own (exact) detections
    fx.pause()
    fx.goto_frame(37)
    qtbot.waitUntil(lambda: fx.current_frame == 37 and rv._last == (37, "exact"), timeout=15000)
    # model toggles and the threshold redraw without re-running anything
    n = len(fx.canvas._dets)
    rv.models_box.itemAt(1).widget().setChecked(False)
    assert all(not b[4].startswith("dummy_b") for b in fx.canvas._dets) and len(fx.canvas._dets) <= n
    rv.conf.setValue(99)
    assert len(fx.canvas._dets) <= 1
    # End review: back to the video open before, at its frame
    rv.end_btn.click()
    assert not rv.active
    qtbot.waitUntil(lambda: fx.video_path.endswith("c.avi") and fx.current_frame == 10, timeout=30000)
    assert fx.canvas._dets == []


def test_opening_another_video_ends_the_review(shell3, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
    from AIO_QC.core.testing.runner import ModelSpec
    fx = shell3.window("video")
    qtbot.addWidget(fx)
    pl = ReviewPlaylist(items=[ReviewItem(os.path.join(D, "a.mp4"))], specs=[ModelSpec(MODELS[0], imgsz=320)],
                        colors=["#e53935"], cache_path=str(tmp_path / "pred.db"))
    rv = shell3.review_in_player(pl)
    qtbot.waitUntil(lambda: fx.video_path and fx.video_path.endswith("a.mp4") and fx.engine is not None,
                    timeout=30000)
    fx.load_video(os.path.join(D, "b.mp4"))
    qtbot.waitUntil(lambda: not rv.active, timeout=30000)
    assert not rv.analysers


def test_invalid_playlist_is_refused(shell3, qtbot, monkeypatch):  # noqa: F811
    from AIO_QC.core.testing.review import ReviewPlaylist
    img = shell3.window("image")
    qtbot.addWidget(img)
    shell3.switch("image")
    assert shell3.review_in_player(ReviewPlaylist()) is None and shell3.mode == "image"


@pytest.mark.parametrize("mode", ["best", "fixed"])
def test_other_modes_play(shell3, qtbot, tmp_path, mode):  # noqa: F811
    from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
    from AIO_QC.core.testing.runner import ModelSpec
    fx = shell3.window("video")
    qtbot.addWidget(fx)
    pl = ReviewPlaylist(items=[ReviewItem(os.path.join(D, "a.mp4"), 0, 2.0)], specs=[ModelSpec(MODELS[0], imgsz=320)],
                        colors=["#e53935"], mode=mode, stride=3, conf=0.05, cache_path=str(tmp_path / "pred.db"))
    rv = shell3.review_in_player(pl)
    qtbot.waitUntil(lambda: fx.is_playing, timeout=30000)
    qtbot.waitUntil(lambda: fx.current_frame > 20 and bool(fx.canvas._dets), timeout=20000)
    st = rv.current().status() if rv.current() else {}
    assert st.get("mode") == mode
    qtbot.waitUntil(lambda: not rv.active, timeout=30000)   # one short video: the playlist ends by itself


# ---------------------------------------------------------------- step 4: events, strip, classes, outputs
def test_events_are_found_and_merged():
    import numpy as np
    from AIO_QC.core.testing.review_events import find_events, pick_hard_frames, summarize
    box = lambda x, c=0.9, k=0: [x, 10, x + 40, 50, c, k]          # noqa: E731
    a, b = {}, {}
    for i in range(0, 100):
        a[i] = np.array([box(10), box(200)], np.float32)
        b[i] = np.array([box(10), box(200)], np.float32)
    for i in (30, 31, 32):                                          # B misses an object: disagree (one event)
        b[i] = np.array([box(10)], np.float32)
    a[60] = np.array([box(10)], np.float32)                         # A loses it for one frame: flicker
    a[80] = np.array([box(10), box(200, 0.2)], np.float32)          # just below 0.25: uncertain
    store = {"A": a, "B": b}
    names = {"A": {0: "car"}, "B": {0: "car"}}
    ev = find_events(store, names, 0, 99, fps=25, thr=0.25, band=0.1)
    kinds = {(e.kind, e.frame) for e in ev}
    assert ("flicker", 60) in kinds and ("uncertain", 80) in kinds
    dis = [e for e in ev if e.kind == "disagree"]
    assert any(e.start == 30 and e.end == 32 for e in dis)          # three frames, one event
    picks = pick_hard_frames(ev, ("disagree", "flicker"), max_n=10, min_gap=5)
    assert [e.frame for e in picks] == [30, 60, 80]                 # one frame per spot, strongest first, in order
    s = summarize({"A": {5: a[5], 60: a[60]}}, 0, 99, 0.25)
    assert s["A"][5] == 2 and s["A"][60] == 1 and s["A"][6] == -1
    # names, not class ids: two models with different class orders agree
    b2 = {i: np.array([box(10, k=3), box(200, k=3)], np.float32) for i in range(10)}
    ev2 = find_events({"A": {i: a[i] for i in range(10)}, "B": b2}, {"A": {0: "car"}, "B": {3: "car"}}, 0, 9, 25)
    assert not [e for e in ev2 if e.kind == "disagree"]


def test_review_panel_events_classes_and_hard_frames(shell3, qtbot, tmp_path, monkeypatch):  # noqa: F811
    import json
    from PyQt6.QtWidgets import QDialog
    from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
    from AIO_QC.core.testing.runner import ModelSpec
    from AIO_QC.tools.frame_extractor.widgets import review_dock as RD
    fx = shell3.window("video")
    qtbot.addWidget(fx)
    fx.output_dir = str(tmp_path / "out")
    pl = ReviewPlaylist(items=[ReviewItem(os.path.join(D, "a.mp4"))], specs=[ModelSpec(p, imgsz=320) for p in MODELS],
                        colors=["#e53935", "#1e88e5"], mode="best", conf=0.3, cache_path=str(tmp_path / "pred.db"))
    rv = shell3.review_in_player(pl)
    qtbot.waitUntil(lambda: rv.current() is not None and rv.current().coverage() >= 1.0, timeout=60000)
    fx.pause()
    qtbot.waitUntil(lambda: bool(rv.events) and rv.ev_list.count() > 0, timeout=20000)
    assert rv.strip.isVisible() and rv.strip.counts and set(rv.strip.counts) <= {s.key for s in pl.specs}
    # events: jump to the next one; clicking the strip seeks
    start = fx.current_frame
    nxt = next((e.frame for e in rv.events if e.frame > start), None)
    if nxt is not None:
        rv.jump_event(1)
        qtbot.waitUntil(lambda: fx.current_frame == nxt, timeout=10000)
    rv.strip.resize(400, 30)
    rv.strip._seek(200)
    qtbot.waitUntil(lambda: abs(fx.current_frame - fx.frame_count // 2) <= 2, timeout=10000)
    # class filter hides a class everywhere
    qtbot.waitUntil(lambda: rv._last[1] == "exact", timeout=10000)
    rv._toggle_class("plate", False)
    assert all(" plate " not in b[4] for b in fx.canvas._dets) and "of" in rv.class_btn.text()
    rv._show_all_classes()
    # CSV of the current video
    csv_path = tmp_path / "d.csv"
    monkeypatch.setattr(RD.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(csv_path), "")))
    rv.export_csv()
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("frame,time_s,model,class") and len(lines) > 100
    # hard frames -> images + LabelMe pre-labels + provenance, opened in Image QC
    opened = []
    monkeypatch.setattr(shell3, "open_in_qc", lambda folder: opened.append(folder) or True)
    monkeypatch.setattr(RD.HardFramesDialog, "exec", lambda self: (self.kinds["uncertain"].setChecked(True),
                                                                   self.max_n.setValue(5),
                                                                   QDialog.DialogCode.Accepted.value)[-1])
    rv.send_hard_frames()
    qtbot.waitUntil(lambda: bool(opened), timeout=30000)
    folder = opened[0]
    imgs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    assert 1 <= len(imgs) <= 5 and all(f.startswith("a_") for f in imgs)
    js = json.loads(open(os.path.join(folder, imgs[0][:-4] + ".json")).read())
    assert js["imagePath"] == imgs[0] and all(s["label"] in ("car", "plate") for s in js["shapes"])
    from AIO_QC.tools.frame_extractor.core.frame_sources import lookup
    src = lookup(os.path.join(folder, imgs[0]))
    assert src and src["video"].endswith("a.mp4") and src["frame"] == int(imgs[0][2:8])
    rv.stop(back=False)
