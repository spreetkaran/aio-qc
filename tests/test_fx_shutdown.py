# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Closing / unloading the Frame Extractor stops every background thread it started (a running QThread destroyed
with its window aborts the whole app)."""
from .test_aio_image import shell  # noqa: F401


def test_unload_stops_roi_tracker(shell, qtbot, monkeypatch):  # noqa: F811
    from AIO_QC.tools.frame_extractor.widgets import track_worker
    from AIO_QC.tools.frame_extractor.widgets.track_worker import TrackWorker

    class Tracker:                                  # stands in for OpenCV's (needs opencv-contrib)
        def __init__(self, kind): pass
        def __getattr__(self, name): return lambda *a, **k: None
    monkeypatch.setattr(track_worker, "RoiTracker", Tracker)
    shell.switch("video")
    v = shell.video
    v.tracker = t = TrackWorker()
    t.start()
    qtbot.waitUntil(t.isRunning, timeout=2000)
    qtbot.wait(200)
    assert t.isRunning()                            # waiting for frames, like during playback
    shell.switch("image")
    shell._unload_video()
    assert shell.video is None
    assert not t.isRunning() and v.__dict__.get("tracker") is None


def test_mpv_terminate_keeps_handle_if_event_thread_hangs(monkeypatch):
    import threading
    from AIO_QC.tools.frame_extractor.core import mpv_lib
    destroyed = []

    class Lib:
        def mpv_wakeup(self, h): pass
        def mpv_terminate_destroy(self, h): destroyed.append(h)
    p = mpv_lib.MpvPlayer.__new__(mpv_lib.MpvPlayer) if hasattr(mpv_lib, "MpvPlayer") else None
    if p is None:
        import pytest
        pytest.skip("no MpvPlayer class")
    stop = threading.Event()
    ev = threading.Thread(target=stop.wait, daemon=True)
    ev.start()
    p.__dict__.update(h=1234, lib=Lib(), _ev=ev, render_ctx=None, on_update=None, _quit=False)
    monkeypatch.setattr(ev, "join", lambda t=None: None)          # the thread does not stop in time
    p.terminate()
    assert destroyed == [] and p.h is None
    stop.set()


def test_studio_shutdown_detaches_a_long_job(shell, qtbot):  # noqa: F811
    """A Data Studio job still running at exit is detached (kept alive until it ends), never destroyed."""
    import threading
    from AIO_QC.widgets.testing.lab_common import Job
    shell.switch("data")
    st = shell.studio
    gate = threading.Event()
    j = Job(lambda *_: gate.wait(20), st)
    j.start()
    qtbot.waitUntil(j.isRunning, timeout=2000)
    st.shutdown()                                               # waits 5 s, then detaches
    assert j.parent() is None and j.isRunning()                 # detached, still alive
    gate.set()
    qtbot.waitUntil(lambda: not j.isRunning(), timeout=5000)
