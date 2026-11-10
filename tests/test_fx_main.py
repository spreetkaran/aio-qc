# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""FrameExtractor on real Qt (offscreen) — audit before integration."""
from __future__ import annotations

import os
import time

import pytest

from .fx_fixtures import D  # noqa: E402  (generated media + models)


@pytest.fixture
def win(qapp, qtbot, tmp_path):
    from AIO_QC.tools.frame_extractor.app import MainWindow
    w = MainWindow()
    w.output_dir = str(tmp_path / "out")
    os.makedirs(w.output_dir, exist_ok=True)
    qtbot.addWidget(w)
    w.show()
    yield w
    w.close()


def _load(w, qtbot, name="a.mp4", fresh=True):
    w.load_video(os.path.join(D, name))
    qtbot.waitUntil(lambda: w.engine is not None and w.current is not None and w.video_path.endswith(name),
                    timeout=30000)
    if fresh and w.markers:                          # sessions persist per video: start each test clean
        w.remove_markers(list(w.markers))
    return w


def test_open_play_seek_close(win, qtbot):
    w = _load(win, qtbot)
    assert w.frame_count > 100
    w.goto_frame(50)
    qtbot.waitUntil(lambda: w.current_frame == 50, timeout=10000)
    w.step_frames(3)
    qtbot.wait(200)
    w.play()
    qtbot.wait(800)
    w.pause()
    assert w.current_frame > 50
    for name in ("b.mp4", "c.avi", "a.mp4"):
        _load(w, qtbot, name)
        qtbot.wait(200)


SKIP = {"quit", "load", "reset", "outdir", "choose_ffmpeg", "choose_libmpv", "mpv", "open_log"}


def _close_tops(w):
    from PyQt6.QtWidgets import QApplication
    for t in QApplication.topLevelWidgets():
        if t is not w and t.isVisible() and t.isWindow():
            t.close()


@pytest.mark.parametrize("video", [False, True])
def test_every_action(win, qtbot, video):
    """Trigger every action with and without a video loaded; nothing may raise."""
    w = win
    if video:
        _load(w, qtbot)
        w.goto_frame(30)
        qtbot.waitUntil(lambda: w.current_frame == 30, timeout=10000)
    done = []
    for name, act in list(vars(w.actions_).items()):
        if name in SKIP or not hasattr(act, "trigger") or not act.isEnabled():
            continue
        t0 = time.monotonic()
        print("ACTION", name, flush=True)
        import faulthandler, sys
        faulthandler.dump_traceback_later(90, exit=True, file=sys.stderr)
        act.trigger()
        qtbot.wait(30)
        if act.isCheckable() and act.isChecked() and name not in ("zoom_fit",):
            act.trigger()                 # toggle back
            qtbot.wait(20)
        _close_tops(w)
        faulthandler.cancel_dump_traceback_later()
        done.append((name, round(time.monotonic() - t0, 2)))
    slow = [d for d in done if d[1] > 2]
    print(len(done), "actions; slow:", slow)
    assert len(done) > (40 if video else 15)


@pytest.fixture
def outdir(win, tmp_path):
    return tmp_path / "out"


def _files(root, pat="*"):
    from pathlib import Path
    return [p for p in Path(root).rglob(pat) if p.is_file()]


def test_markers_save_roi_timeline(win, qtbot, outdir):
    w = _load(win, qtbot)
    qtbot.waitUntil(lambda: len(w.strip.samples) > 5, timeout=30000)            # timeline analysis
    for f in (10, 40, 90):
        w.goto_frame(f)
        qtbot.waitUntil(lambda f=f: w.current is not None and w.current.idx == f and not w.current.preview,
                        timeout=10000)
        w.toggle_marker()
    assert len(w.markers) == 3
    w.extract_markers()
    qtbot.waitUntil(lambda: all(m.get("saved") for m in w.markers.values()), timeout=30000)
    assert len(_files(outdir / "a" / "markers", "*.jpg")) >= 3
    added = w.suggest_markers("sharpest", 2.0)
    assert added
    w.snap_markers(added, 5, then_extract=True)
    qtbot.waitUntil(lambda: all(m.get("saved") for m in w.markers.values()), timeout=60000)
    w.auto_pick()
    qtbot.wait(300)
    # save frames: full and ROI crop
    w.on_roi_changed((100, 60, 300, 200))
    w.save_frame()
    w.save_frame(full=True)
    qtbot.wait(100)
    shots = [p for p in _files(outdir / "a") if "markers" not in p.parts]
    assert len(shots) >= 2, shots
    w.send_markers_to_batch()
    assert w.batch_dialog.table.rowCount() == 1
    w.review_markers()
    qtbot.wait(200)


def test_detections_and_tracking(win, qtbot, outdir):
    w = _load(win, qtbot)
    w.model_cfg["models"] = [f"{D}/models/dummy_a.onnx", f"{D}/models/dummy_b.onnx"]
    w.actions_.detect.setChecked(True)
    w.toggle_detections(True)
    qtbot.waitUntil(lambda: bool(w.detections_cache), timeout=30000)
    w.goto_frame(60)
    qtbot.waitUntil(lambda: 60 in w.detections_cache, timeout=30000)
    w.toggle_detections(False)
    w.on_roi_changed((200, 100, 300, 180))
    w.actions_.track.setChecked(True)
    w.toggle_tracking(True)
    w.play()
    qtbot.wait(800)
    w.pause()
    w.toggle_tracking(False)
    qtbot.wait(100)


def test_batch_queue_runs(win, qtbot, outdir):
    w = _load(win, qtbot)
    w.set_in()
    w.goto_frame(120)
    qtbot.waitUntil(lambda: w.current_frame == 120, timeout=10000)
    w.set_out()
    w.open_batch_queue()
    b = w.batch_dialog
    b.add_job(f"{D}/b.mp4")
    b.add_job(f"{D}/c.avi")
    assert b.table.rowCount() == 3
    b.start()
    qtbot.waitUntil(lambda: not b.is_running(), timeout=180000)
    stat = [b._row_status(r) for r in range(b.table.rowCount())]
    assert all(s.startswith("Done") for s in stat), stat
    assert len(_files(outdir, "*.jpg")) > 5
    b.close()


def test_exports_gallery_dialogs(win, qtbot, outdir, tmp_path):
    from AIO_QC.tools.frame_extractor.widgets.export_dialogs import AnnotatedExportDialog, ClipExportDialog
    from AIO_QC.core.testing import pipeline as P
    w = _load(win, qtbot)
    ff = w._ffmpeg()
    assert ff
    for mode in ("fast", "exact"):
        d = ClipExportDialog(w, w.video_path, 1.0, 3.0, w.engine.duration(), ff)
        qtbot.addWidget(d)
        (d.fast if mode == "fast" else d.exact).setChecked(True)
        d.out_edit.setText(str(tmp_path / f"clip_{mode}.mp4"))
        d.start()
        qtbot.waitUntil(lambda: d.job is None, timeout=60000)
        assert d.status.text().startswith("Saved"), d.status.text()
        d.close()
    specs = P.unique_labels([P.spec_from_cfg(m, {"models": [], "conf": 0.25, "imgsz": 0, "iou": 0.7, "device": "cpu",
                                                  "env_python": "", "half": False, "sahi": False})
                             for m in (f"{D}/models/dummy_a.onnx", f"{D}/models/dummy_b.onnx")])
    d = AnnotatedExportDialog(w, w.video_path, specs, ff, 0.0, 2.0, w.engine.duration())
    qtbot.addWidget(d)
    d.out_edit.setText(str(tmp_path / "annotated.mp4"))
    d.start()
    qtbot.waitUntil(lambda: d.job is None, timeout=120000)
    assert d.status.text().startswith("Saved"), d.status.text()
    d.close()
    # gallery: reject / restore / undo on extracted frames
    w.goto_frame(20)
    qtbot.waitUntil(lambda: w.current_frame == 20 and not w.current.preview, timeout=10000)
    for f in (20, 60, 100):
        w.goto_frame(f)
        qtbot.waitUntil(lambda f=f: w.current.idx == f and not w.current.preview, timeout=10000)
        w.save_frame(full=True)
    w.open_gallery(str(outdir / "a"))
    g = w.gallery
    qtbot.waitUntil(lambda: g.list.count() >= 3, timeout=10000)
    n = g.list.count()
    g.list.setCurrentRow(0)
    g.list.item(0).setSelected(True)
    g.reject_selected()
    assert g.list.count() == n - 1
    g.undo_last()
    assert g.list.count() == n
    g.close()
    # other windows: help, get started, palette, shortcuts, output, models
    w.open_help()
    w.show_get_started()
    w.show_shortcuts()
    w.open_output_settings()
    w.open_models()
    qtbot.wait(200)


def test_settings_switches(win, qtbot):
    w = _load(win, qtbot)
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QColor, QPalette
    app = QApplication.instance()
    old = app.palette()
    dark = QPalette(old)
    dark.setColor(QPalette.ColorRole.Window, QColor("#202124"))
    for pal in (dark, old):                     # one theme: AIO QC's palette; the window repaints on changes
        app.setPalette(pal)
        qtbot.wait(30)
    assert not hasattr(w, "theme_actions")
    for r in ("software", "auto"):
        w.set_renderer(r)
        qtbot.wait(50)
    for b in ("opencv", "pyav", "auto"):
        w.set_backend(b)
        _load(w, qtbot)
    w.set_playback_mode("off")
    w.play()
    qtbot.wait(300)
    w.pause()
