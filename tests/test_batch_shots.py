# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Batch screenshots: Review ▸ Flagged Screenshots and File list ▸ Save screenshots of N files."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QImage

from AIO_QC.core.state import REVIEW_FLAG
from AIO_QC.widgets.batch_shots import BatchShots

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _win(shell, qtbot, tmp_path, n=6):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", n=n))
    qtbot.waitUntil(lambda: w.current_key is not None and bool(w.canvas.shape_groups), timeout=10000)
    w.resize(1100, 800)
    w.settings.setValue("screenshot/dir", str(tmp_path / "shots"))
    w.settings.setValue("screenshot/format", "png")
    qtbot.wait(50)
    return w


def _flag(w, keys, note="plate text is wrong — check the 5th character"):
    w.session.set_review(keys, REVIEW_FLAG, note)


def _run(qtbot, job, timeout=20000):
    qtbot.waitUntil(lambda: job.finished, timeout=timeout)
    return job


def test_flagged_same_spot_with_caption(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    keys = [s.key for s in w.session.samples]
    flagged = keys[1:4]
    _flag(w, flagged)
    c = w.canvas
    c.animate_to(3.0, center=QPointF(90, 80), duration=0)
    z, current = c.current_zoom, w.current_key
    from AIO_QC.widgets.screenshot import capture_view
    want = capture_view(c)                                               # what E would save right now
    dlg = w.open_batch_shots()
    assert dlg is not None and "All flagged files (3)" in dlg.scope.currentText()
    assert dlg.view_btns["spot"].isChecked() and dlg.caption.isChecked() and dlg.sketch.isChecked()   # defaults
    job = _run(qtbot, dlg.start())
    assert job.written == 3 and not job.errors
    for k in flagged:
        p = tmp_path / "shots" / "ds" / f"{Path(k).stem}.png"
        assert p.is_file()
        img = QImage(str(p))
        assert img.width() == want.width()                                  # the zoomed-in view, as on screen
        assert img.height() > want.height()                                 # + the caption strip
        strip = img.pixelColor(2, img.height() - 2)
        assert (strip.red(), strip.green(), strip.blue()) == (28, 28, 30)
    # the window's own view was never touched
    assert w.current_key == current and abs(c.current_zoom - z) < 1e-9
    assert c.pixmap_item.pixmap().cacheKey() == w.pixmap_cache.get(current).cacheKey()      # its own image still
    assert w.toast.find("batch_shots") is not None


def test_whole_image_no_caption_and_existing_files(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    keys = [s.key for s in w.session.samples][:2]
    job = _run(qtbot, BatchShots(w, keys, "whole", caption=False, sketch=False, existing="keep").start())
    assert job.written == 2
    out = tmp_path / "shots" / "ds" / f"{Path(keys[0]).stem}.png"
    img = QImage(str(out))
    dpr = w.canvas.viewport().devicePixelRatioF()
    assert abs(img.width() - 320 * (1 if dpr == 1 else 1)) <= 1 and abs(img.height() - 240) <= 1   # full resolution
    # keep both / skip / overwrite
    job = _run(qtbot, BatchShots(w, keys[:1], "whole", False, False, "keep").start())
    assert (out.parent / f"{out.stem}_2.png").is_file()
    job = _run(qtbot, BatchShots(w, keys[:1], "whole", False, False, "skip").start())
    assert job.skipped == 1 and job.written == 0
    before = out.stat().st_mtime_ns
    job = _run(qtbot, BatchShots(w, keys[:1], "whole", True, False, "overwrite").start())
    assert job.written == 1 and out.stat().st_mtime_ns != before and QImage(str(out)).height() > 240


def test_sketch_and_own_brightness_are_drawn(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets.sketch import SketchStore
    w = _win(shell, qtbot, tmp_path)
    key = w.session.samples[2].key
    SketchStore(w.session.root).save(key, [{"t": "rect", "c": "#ff00ff", "w": 8, "pts": [[20, 20], [200, 200]]}])
    with_sk = _run(qtbot, BatchShots(w, [key], "whole", False, True, "keep").start()).paths[0]
    without = _run(qtbot, BatchShots(w, [key], "whole", False, False, "keep").start()).paths[0]
    a, b = QImage(str(with_sk)), QImage(str(without))
    px = a.pixelColor(20, 100)
    assert (px.red(), px.green(), px.blue()) == (255, 0, 255) and b.pixelColor(20, 100) != px
    # a file's own brightness from this session
    w._display_by_key[key] = {"brightness": 90, "contrast": 0, "gamma": 100, "auto": False}
    bright = QImage(str(_run(qtbot, BatchShots(w, [key], "whole", False, False, "keep").start()).paths[0]))
    q1, q2 = bright.pixelColor(300, 230), b.pixelColor(300, 230)
    assert q1.red() + q1.green() + q1.blue() > q2.red() + q2.green() + q2.blue() + 100


def test_file_list_entry_cancel_and_missing(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path, n=12)
    keys = [s.key for s in w.session.samples]
    dlg = w.open_batch_shots(keys[:5])
    assert dlg.scope.currentText().startswith("Selected files (5)")
    job = BatchShots(w, keys, "spot", True, True, "keep")
    job.start()
    job.cancel()
    _run(qtbot, job)
    assert job.cancelled and job.written < len(keys)
    # a file that left the dataset is reported, the rest is saved
    job = _run(qtbot, BatchShots(w, [keys[0], str(tmp_path / "gone.json")], "whole", False, False, "keep").start())
    assert job.written == 1 and len(job.errors) == 1
    # no flagged files: nothing to start
    dlg.set_keys(None)
    assert dlg.scope.currentText().startswith("All flagged files (0)") and not dlg.start_btn.isEnabled()
