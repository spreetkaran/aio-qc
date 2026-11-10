# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Three modes in one app: switching, lazy loading, shortcuts that never collide inside a window, quitting."""
from __future__ import annotations

import os
import sys

import pytest

from .fx_fixtures import D


def shortcut_table(win):
    """{key sequence: [action names]} of every shortcut active in this window (actions + QShortcut objects)."""
    from PyQt6.QtGui import QAction, QShortcut
    from PyQt6.QtCore import Qt
    out = {}
    seen = set()
    for a in win.findChildren(QAction):
        if id(a) in seen or a.shortcutContext() == Qt.ShortcutContext.WidgetShortcut:
            continue
        seen.add(id(a))
        for ks in a.shortcuts():
            k = ks.toString()
            if k:
                out.setdefault(k, []).append(a.text().replace("&", "").replace("\n", " ") or a.objectName())
    for s in win.findChildren(QShortcut):
        if s.context() == Qt.ShortcutContext.WidgetShortcut or not s.isEnabled():
            continue
        k = s.key().toString()
        if k:
            out.setdefault(k, []).append(f"QShortcut:{s.objectName() or '?'}")
    return out


def clashes(win):
    """Keys bound to two DIFFERENT commands in one window (Qt fires neither: 'Ambiguous shortcut overload')."""
    return {k: v for k, v in shortcut_table(win).items() if len(set(v)) > 1 or len(v) > 1}


@pytest.fixture
def shell3(qapp, qtbot):
    from AIO_QC.shell import AppShell
    sh = AppShell(qapp)
    qapp.shell = sh
    qapp.main_window = sh
    yield sh
    sh._closing = True
    if sh.video is not None:
        sh.video.close()
    if sh.studio is not None:
        sh.studio.shutdown()
        sh.studio.close()
    if sh.image is not None:
        sh.image._quit_reason = "test"
        sh.image.close()
    qapp.shell = None


def test_mode_switching_smoke(shell3, qtbot):
    sh = shell3
    w = sh.show("image")
    qtbot.addWidget(w)
    order = []
    for _ in range(7):                                  # Ctrl+Alt+M cycles through all three, twice and more
        sh.cycle()
        qtbot.wait(30)
        order.append(sh.mode)
        cur = sh.window(sh.mode)
        assert cur.isVisible()
        assert sum(x is not None and x.isVisible() for x in (sh.image, sh.video, sh.studio)) == 1
        assert cur.mode_switch.current == sh.mode
    assert order[:3] == ["data", "video", "image"]      # order: Frame Extractor, Image QC, Data Studio
    # the Frame Extractor keeps its video and position across switches, and pauses when hidden
    sh.switch("video")
    v = sh.video
    v.load_video(os.path.join(D, "a.mp4"))
    qtbot.waitUntil(lambda: v.engine is not None and v.current is not None, timeout=30000)
    v.goto_frame(40)
    qtbot.waitUntil(lambda: v.current_frame == 40, timeout=10000)
    v.play()
    qtbot.wait(200)
    sh.switch("data")
    assert not v.is_playing
    sh.switch("video")
    assert v.video_path.endswith("a.mp4") and v.current_frame >= 40
    # every window offers the other modes in its File menu
    for win in (sh.image, sh.video, sh.studio):
        texts = [a.text() for a in win.menuBar().actions()[0].menu().actions()]
        assert sum(t.startswith("Switch to") for t in texts) == 2, texts
        assert any(t.startswith("Next") for t in texts), texts


def test_no_shortcut_clashes_in_any_window(shell3, qtbot):
    sh = shell3
    for m in ("image", "video", "data"):
        sh.switch(m) if sh.mode else sh.show(m)
        qtbot.wait(30)
    bad = {m: clashes(sh.window(m)) for m in ("image", "video", "data")}
    bad = {m: c for m, c in bad.items() if c}
    assert not bad, bad


def test_quit_all_closes_every_mode(shell3, qtbot, qapp):
    sh = shell3
    sh.show("video")
    sh.switch("image")
    sh.switch("data")
    quit_called = []
    qapp.quit = lambda: quit_called.append(1)
    sh.quit_all()
    assert quit_called
    assert not sh.video.isVisible()


def _live_state():
    import threading
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QApplication
    from AIO_QC.core import procs
    running_qthreads = [t for t in QApplication.instance().findChildren(QThread) if t.isRunning()]
    children = [p for p in list(procs._live) if p.poll() is None]
    threads = {t.name for t in threading.enumerate() if t.is_alive() and not t.daemon}
    return running_qthreads, children, threads


def test_tool_open_load_extract_close_no_leaks(shell3, qtbot, tmp_path):
    import gc
    from pathlib import Path
    sh = shell3
    sh.show("image")
    base_threads = _live_state()[2]
    sh.switch("video")
    v = sh.video
    v.output_dir = str(tmp_path / "out")
    v.load_video(os.path.join(D, "a.mp4"))
    qtbot.waitUntil(lambda: v.engine is not None and v.current is not None, timeout=30000)
    qtbot.waitUntil(lambda: len(v.strip.samples) > 3, timeout=30000)
    for f in (10, 50):
        v.goto_frame(f)
        qtbot.waitUntil(lambda f=f: v.current.idx == f and not v.current.preview, timeout=10000)
        v.toggle_marker()
    v.extract_markers()
    qtbot.waitUntil(lambda: all(m.get("saved") for m in v.markers.values()), timeout=30000)
    v.play()
    qtbot.wait(300)
    v.pause()
    assert len(list(Path(v.output_dir).rglob("*.jpg"))) >= 2
    sh.switch("image")
    v.close()                                       # what quitting does to this window
    qtbot.wait(300)
    gc.collect()
    qthreads, children, threads = _live_state()
    assert not qthreads, [type(t).__name__ for t in qthreads]
    assert not children, children
    assert threads <= base_threads | {"MainThread"}, threads - base_threads
    assert v.engine is None and v.marker_worker is None and v.detector is None and v.timeline is None


def test_cli_subcommands_dispatch(tmp_path):
    import subprocess
    root = str(Path_root())
    env = dict(os.environ, PYTHONPATH=root, HOME=str(tmp_path))
    r = subprocess.run([sys.executable, "-m", "AIO_QC", "probe", os.path.join(D, "c.avi")], capture_output=True,
                       text=True, env=env, cwd=root, timeout=120)
    assert r.returncode == 0 and "320x240" in r.stdout, r.stderr
    r = subprocess.run([sys.executable, "-m", "AIO_QC", "extract", os.path.join(D, "c.avi"), "-o", str(tmp_path / "o"),
                        "-n", "15"], capture_output=True, text=True, env=env, cwd=root, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert len(list((tmp_path / "o").rglob("*.jpg"))) >= 3
    r = subprocess.run([sys.executable, "-m", "AIO_QC.tools.frame_extractor", "probe", os.path.join(D, "a.mp4")],
                       capture_output=True, text=True, env=env, cwd=root, timeout=120)
    assert r.returncode == 0 and "640x360" in r.stdout


def Path_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


def test_mode_order_default_and_icon(shell3, qtbot):
    from AIO_QC import shell as S
    from AIO_QC.widgets.mode_switch import ModeSwitch  # noqa: F401
    assert S.MODES[0] == "video" and S.MODES[1] == "image"
    sh = shell3
    assert sh.start_mode(None) == "image" and sh.start_mode("bogus") == "image" and sh.start_mode("video") == "video"
    w = sh.show("image")
    qtbot.addWidget(w)
    v = sh.window("video")
    ic = v.windowIcon()
    assert not ic.isNull()
    a = ic.pixmap(32, 32).toImage()
    b = w.windowIcon().pixmap(32, 32).toImage()
    assert a != b                                       # the Frame Extractor keeps its own app icon


def test_switching_to_frame_extractor_never_fades_its_gl_window(shell3, qtbot, monkeypatch):
    """A window opacity makes a window layered on Windows, and the Frame Extractor's OpenGL canvas fails there
    (the app then remembers 'GPU failed' and stays on CPU). The video window must never be faded."""
    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from AIO_QC.widgets import anim, motion
    calls = []
    orig = QWidget.setWindowOpacity

    def spy(self, x):
        calls.append(self)
        return orig(self, x)
    monkeypatch.setattr(QWidget, "setWindowOpacity", spy)
    sh = shell3
    w = sh.show("image")
    qtbot.addWidget(w)
    for m in ("video", "image", "video", "data", "video"):
        sh.switch(m)
        qtbot.wait(60)
    assert sh.video is not None and all(c is not sh.video for c in calls)
    host = QWidget()
    qtbot.addWidget(host)
    QOpenGLWidget(host)
    assert anim.hosts_opengl(host)
    calls.clear()
    anim.window_fade_in(host)
    motion.fade_window(host)
    assert host not in calls
