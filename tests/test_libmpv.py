# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""libmpv: loading and diagnostics, the start-up order (render context before the file), the picture check with its
basic-renderer fallback, the playback-rate guard, the libmpv dialog, the packaging helper - and, when a real libmpv
and an X server are available, a real playback run (AIOQC_TEST_LIBMPV=/path/to/libmpv)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .fx_fixtures import D

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- later(): every call schedules a new timer
def test_later_fires_every_time_for_the_same_callable(qapp, qtbot):
    """later() was memoised (lru_cache): a second call with the same bound method returned the first, already
    finished timer - it never fired again - and every owner / lambda passed in was kept alive forever."""
    from PyQt6.QtWidgets import QWidget
    from AIO_QC.utils.qt_utils import later
    w = QWidget()
    qtbot.addWidget(w)
    hits = []

    class Target:
        def tick(self):
            hits.append(1)
    t = Target()
    a = later(0, w, t.tick)
    qtbot.waitUntil(lambda: len(hits) == 1, timeout=2000)
    b = later(0, w, t.tick)
    assert b is not a
    qtbot.waitUntil(lambda: len(hits) == 2, timeout=2000)
    assert not hasattr(later, "cache_info")


# ---------------------------------------------------------------- mpv_lib: loading and diagnostics
def test_probe_reports_the_real_reason(tmp_path):
    from AIO_QC.tools.frame_extractor.core import mpv_lib
    bad = tmp_path / ("libmpv-2.dll" if sys.platform == "win32" else "libmpv.so.2")
    bad.write_bytes(b"not a library")
    lib, msg = mpv_lib.probe(str(bad))
    assert lib is None and str(bad) in msg
    assert "undefined symbol" in mpv_lib._hint("x: undefined symbol: g_string_copy") or \
        "newer versions" in mpv_lib._hint("x: undefined symbol: g_string_copy")
    mpv_lib.reset()
    assert mpv_lib.load_library(str(tmp_path / "gone.so")) is None or mpv_lib._LIB is not None
    if mpv_lib._LIB is None:
        assert "no longer exists" in mpv_lib.last_error()
    mpv_lib.reset()
    if mpv_lib.load_library(str(bad)) is None:
        assert str(bad) in mpv_lib.last_error()           # the chosen file's own error, not "not found"
    mpv_lib.reset()


def test_search_dirs_and_diagnose(monkeypatch, tmp_path):
    from AIO_QC.tools.frame_extractor.core import mpv_lib
    monkeypatch.setenv("FX_LIBMPV_DIR", str(tmp_path))
    dirs = [d for d, _ in mpv_lib.search_dirs()]
    assert dirs[0] == str(tmp_path)
    assert any(d.endswith(os.path.join("frame_extractor", "lib")) for d in dirs)
    assert mpv_lib.user_lib_dir() in dirs
    rows = mpv_lib.diagnose("")
    assert rows and all(os.path.isabs(p) for p, _ in rows)


def test_pending_dll_is_put_in_place(monkeypatch, tmp_path):
    from AIO_QC.tools.frame_extractor.core import mpv_lib
    monkeypatch.setattr(mpv_lib, "user_lib_dir", lambda: str(tmp_path))
    (tmp_path / "libmpv-2.dll.new").write_bytes(b"new")
    mpv_lib._promote_pending()
    assert (tmp_path / "libmpv-2.dll").read_bytes() == b"new" and not (tmp_path / "libmpv-2.dll.new").exists()


# ---------------------------------------------------------------- the dialog
def test_install_instructions_per_os():
    from AIO_QC.tools.frame_extractor.widgets import libmpv_dialog as L
    assert L.linux_command('ID=ubuntu\nVERSION_ID="22.04"').startswith("sudo apt install libmpv2")
    assert "dnf" in L.linux_command("ID=fedora")
    assert "pacman" in L.linux_command('ID="manjaro"\nID_LIKE=arch')
    assert "zypper" in L.linux_command('ID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"')
    assert "apt" in L.linux_command('ID=linuxmint\nID_LIKE="ubuntu debian"')
    html, cmd, url = L.instructions("win32")
    assert "mpv-dev-lgpl" in html and cmd == "" and "mpv-winbuild" in url
    html, cmd, url = L.instructions("darwin")
    assert cmd == "brew install mpv"
    html, cmd, url = L.instructions("linux")
    assert cmd


def test_libmpv_dialog_opens_from_the_playback_menu(qapp, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    from AIO_QC.tools.frame_extractor.app import MainWindow
    from AIO_QC.tools.frame_extractor.widgets import libmpv_dialog as L
    w = MainWindow()
    qtbot.addWidget(w)
    seen = []
    monkeypatch.setattr(QDialog, "exec", lambda self: seen.append(self) or 0)
    act = next(a for a in w.findChildren(__import__("PyQt6.QtGui", fromlist=["QAction"]).QAction)
               if a.objectName() == "playback_libmpv")
    act.trigger()
    assert seen and isinstance(seen[0], L.LibmpvDialog)
    dlg = seen[0]
    qtbot.addWidget(dlg)
    assert dlg.status.text()
    dlg.more.setChecked(True)
    assert dlg.details.toPlainText()
    w.close()


# ---------------------------------------------------------------- start-up order, picture check, rate guard
class FakePlayer:
    log: list = []
    made: list = []

    def __init__(self, lib, hwdec="auto-safe", log_level="warn", basic=False):
        self.basic = basic
        self.loaded, self.failed, self.restarts, self.t0, self.path = False, None, 0, 0.0, None
        self.render_ctx, self.closing, self.on_update, self.pos = None, False, None, None
        self.props = {}
        FakePlayer.made.append(self)

    def load(self, path, aspect=None):
        FakePlayer.log.append("load")
        self.path = path
        return True

    def set(self, k, v):
        self.props[k] = v
        return 0

    def set_double(self, k, v):
        self.props[k] = v

    def set_display_fps(self, hz):
        pass

    def get_double(self, k):
        return 30.0

    def get_flag(self, k):
        return False

    def get_str(self, k):
        return "no"

    @property
    def time_pos(self):
        return self.pos

    def seek(self, *a):
        pass

    def play(self):
        pass

    def pause(self):
        pass

    def command(self, *a):
        return 0

    def terminate(self):
        FakePlayer.log.append("terminate")


@pytest.fixture
def fxw(qapp, qtbot, tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.app import MainWindow
    from AIO_QC.tools.frame_extractor.core import mpv_lib
    w = MainWindow()
    w.output_dir = str(tmp_path)
    qtbot.addWidget(w)
    w.show()
    w.load_video(os.path.join(D, "a.mp4"))
    qtbot.waitUntil(lambda: w.engine is not None and w.current is not None and w.current.bgr is not None,
                    timeout=30000)
    qtbot.wait(700)              # let the window's own delayed start-up calls (timeline after 600 ms) run now
    FakePlayer.log, FakePlayer.made = [], []
    monkeypatch.setattr(mpv_lib, "load_library", lambda *a, **k: object())
    monkeypatch.setattr(mpv_lib, "MpvPlayer", FakePlayer)
    c = w.canvas
    monkeypatch.setattr(c, "gpu", True, raising=False)
    monkeypatch.setattr(c, "attach_mpv", lambda m: setattr(c, "_mpv", m), raising=False)
    monkeypatch.setattr(c, "release_mpv", lambda: setattr(c, "_mpv", None), raising=False)
    w.settings.setValue("playback/engine", "mpv")
    yield w
    w.stop_mpv()
    w.settings.setValue("playback/engine", "auto")
    w.close()
    qtbot.wait(300)              # pending verify polls / timers fire while the window still exists


def test_render_context_exists_before_the_file_is_loaded(fxw, qtbot, monkeypatch):
    """vo=libmpv opens its video output only while a render context exists: loading first failed with
    'no audio or video data played' (the libmpv-2.dll report)."""
    w, c = fxw, fxw.canvas
    ready = iter([False, False, True])

    def prepare():
        ok = next(ready, True)
        FakePlayer.log.append("render" if ok else "not-ready")
        return ok
    monkeypatch.setattr(c, "prepare_mpv", prepare, raising=False)
    monkeypatch.setattr(c, "verify_mpv", lambda bgr: None, raising=False)
    w._configure_mpv()
    qtbot.waitUntil(lambda: "load" in FakePlayer.log, timeout=3000)
    assert FakePlayer.log[:4] == ["not-ready", "not-ready", "render", "load"]


def _ready_player(w):
    m = w.mpv
    m.loaded, m.restarts, m.pos = True, 1, w.engine.time_of(w.current.idx)
    return m


def test_wrong_picture_retries_with_the_basic_renderer(fxw, qtbot, monkeypatch):
    w, c = fxw, fxw.canvas
    answers = iter([False, True])
    monkeypatch.setattr(c, "prepare_mpv", lambda: True, raising=False)
    monkeypatch.setattr(c, "verify_mpv", lambda bgr: next(answers), raising=False)
    w._mpv_verified = w._mpv_basic = False
    w._configure_mpv()
    _ready_player(w)
    w._mpv_verify()
    assert w._mpv_basic and FakePlayer.made[-1].basic and "terminate" in FakePlayer.log
    _ready_player(w)
    w._mpv_verify()
    assert w._mpv_verified and w.mpv is not None
    assert "verified" in w.mpv_state()["state"] and "basic" in w.mpv_state()["state"]


def test_wrong_picture_twice_turns_libmpv_off(fxw, qtbot, monkeypatch):
    w, c = fxw, fxw.canvas
    monkeypatch.setattr(c, "prepare_mpv", lambda: True, raising=False)
    monkeypatch.setattr(c, "verify_mpv", lambda bgr: False, raising=False)
    w._mpv_verified = w._mpv_basic = False
    w._configure_mpv()
    _ready_player(w)
    w._mpv_verify()
    _ready_player(w)
    w._mpv_verify()
    assert w.mpv is None and w.settings.value("playback/engine") == "builtin"


def test_flat_frame_is_not_a_verdict(fxw, monkeypatch):
    w, c = fxw, fxw.canvas
    monkeypatch.setattr(c, "prepare_mpv", lambda: True, raising=False)
    monkeypatch.setattr(c, "verify_mpv", lambda bgr: None, raising=False)
    w._mpv_verified = w._mpv_basic = False
    w._configure_mpv()
    _ready_player(w)
    w._mpv_verify()
    assert not w._mpv_verified and not w._mpv_basic and w.mpv is not None


def test_rate_guard_switches_to_clock_sync(fxw, monkeypatch):
    """display-resample follows the canvas' swaps: if they don't come at the display rate the video plays slow /
    fast - then mpv is switched to its own clock (and stays so for the next videos)."""
    from AIO_QC.tools.frame_extractor import app as A
    w, c = fxw, fxw.canvas
    monkeypatch.setattr(c, "prepare_mpv", lambda: True, raising=False)
    monkeypatch.setattr(c, "verify_mpv", lambda bgr: None, raising=False)
    monkeypatch.setattr(type(c), "mpv_showing", property(lambda self: True), raising=False)
    w._mpv_clock_sync = False
    w._configure_mpv()
    m = w.mpv
    now = [1000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: now[0])
    w._mpv_t_play, w._mpv_rate_ref = 999.0, None
    w._mpv_check_rate(m, 1.0)                     # reference taken
    now[0] += 2.0
    w._mpv_check_rate(m, 2.0)                     # 1 s of video in 2 s: 0.5x
    assert w._mpv_clock_sync and m.props.get("video-sync") == "audio"
    w._mpv_clock_sync = False
    w._mpv_rate_ref = (now[0], 2.0)
    m.props.clear()
    now[0] += 2.0
    w._mpv_check_rate(m, 4.0)                     # 1.0x: left alone
    assert not w._mpv_clock_sync and "video-sync" not in m.props


# ---------------------------------------------------------------- packaging helper (pure parts)
def _fetch():
    sys.path.insert(0, str(ROOT / "packaging"))
    import fetch_libmpv
    return fetch_libmpv


def test_windows_asset_choice():
    F = _fetch()
    rel = {"assets": [{"name": n, "size": 1} for n in (
        "mpv-dev-lgpl-x86_64-20260821-git-49418246f3.7z", "mpv-dev-lgpl-aarch64-20260821-git-49418246f3.7z",
        "mpv-dev-x86_64-20260821-git-49418246f3.7z", "mpv-dev-x86_64-v3-20260821-git-49418246f3.7z",
        "mpv-x86_64-20260821-git-49418246f3.7z", "ffmpeg-lgpl-x86_64-git-40000d0f5.7z")]}
    assert F.choose_windows_asset(rel, "x86_64", False)["name"].startswith("mpv-dev-lgpl-x86_64-2026")
    assert F.choose_windows_asset(rel, "aarch64", False)["name"].startswith("mpv-dev-lgpl-aarch64-")
    assert F.choose_windows_asset(rel, "x86_64", True)["name"] == "mpv-dev-x86_64-20260821-git-49418246f3.7z"
    assert "-v3-" in F.choose_windows_asset(rel, "x86_64", True, v3=True)["name"]
    assert F.choose_windows_asset(rel, "aarch64", True) is None


def test_linux_and_macos_closure_parsers():
    F = _fetch()
    deps = F.parse_ldd("\tlinux-vdso.so.1 (0x00007ffd)\n"
                       "\tlibavcodec.so.58 => /lib/x86_64-linux-gnu/libavcodec.so.58 (0x00007f)\n"
                       "\tlibglib-2.0.so.0 => /lib/x86_64-linux-gnu/libglib-2.0.so.0 (0x00007f)\n"
                       "\tlibfoo.so.1 => not found\n\t/lib64/ld-linux-x86-64.so.2 (0x00007f)")
    assert deps == {"libavcodec.so.58": "/lib/x86_64-linux-gnu/libavcodec.so.58",
                    "libglib-2.0.so.0": "/lib/x86_64-linux-gnu/libglib-2.0.so.0", "libfoo.so.1": None}
    sysl = ("libc.so.6", "libglib-2.0.so.0", "libX11.so.6", "libxcb-shm.so.0", "libstdc++.so.6", "libva.so.2",
            "libva-drm.so.2", "libGL.so.1", "libasound.so.2", "ld-linux-x86-64.so.2", "libfontconfig.so.1")
    ours = ("libavcodec.so.58", "libass.so.9", "libplacebo.so.338", "libvapoursynth-script.so.0", "libx264.so.164")
    assert all(F.linux_is_system(n) for n in sysl)
    assert not any(F.linux_is_system(n) for n in ours)
    out = F.parse_otool("/x/libmpv.2.dylib:\n"
                        "\t@rpath/libmpv.2.dylib (compatibility version 2.0.0, current version 2.5.0)\n"
                        "\t@rpath/libavcodec.62.dylib (compatibility version 62.0.0, current version 62.1.100)\n"
                        "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1345.0.0)")
    assert out == ["@rpath/libmpv.2.dylib", "@rpath/libavcodec.62.dylib", "/usr/lib/libSystem.B.dylib"]
    assert F.mac_is_system(out[2]) and not F.mac_is_system(out[1])


def test_spec_ships_libraries_as_binaries():
    spec = (ROOT / "packaging" / "aio_qc.spec").read_text(encoding="utf-8")
    assert "binaries=binaries" in spec and '"lib")]' not in spec


# ---------------------------------------------------------------- real libmpv (opt-in)
REAL = os.environ.get("AIOQC_TEST_LIBMPV", "")


@pytest.mark.skipif(not (REAL and os.path.isfile(REAL) and shutil.which("xvfb-run")),
                    reason="set AIOQC_TEST_LIBMPV=/path/to/libmpv (and have xvfb-run) for the real playback run")
def test_real_libmpv_playback(tmp_path):
    """Opens, verifies the picture, plays, pauses on the exact frame, steps, pauses when hidden, reaches the end
    and plays a second video - all through libmpv on a real OpenGL context (Xvfb + Mesa)."""
    env = dict(os.environ, LIBMPV=REAL, HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "cfg"),
               XDG_DATA_HOME=str(tmp_path / "data"), AIOQC_ROOT=str(ROOT), MEDIA=D)
    env.pop("QT_QPA_PLATFORM", None)
    extra = os.environ.get("AIOQC_TEST_LD_LIBRARY_PATH", "")
    if extra:
        env["LD_LIBRARY_PATH"] = extra + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    r = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", sys.executable,
                        str(Path(__file__).with_name("libmpv_scenario.py"))],
                       env=env, capture_output=True, text=True, timeout=180)
    res = dict(line[2:].split(" => ", 1) for line in r.stdout.splitlines() if line.startswith("R "))
    assert res, r.stdout[-3000:] + r.stderr[-3000:]
    assert res["gpu"] == "GPU"
    assert res["mpv loaded"] == "True" and res["verified"] == "True"
    assert res["mpv playing"] == "True" and res["frame advanced"] == "True"
    assert res["mpv picture not black"] == "True"
    assert res["paused exact"] == "True" and res["step +1"] == "True"
    assert res["paused when hidden"] == "True" and res["reached end"] == "True"
    assert res["second video via mpv"] == "True" and res["failure"] == "None"


# ---------------------------------------------------------------- automatic download / install (core.mpv_fetch)
def test_fetch_plan_per_os():
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    assert F.plan("win32")["how"] == "download" and "LGPL" in F.plan("win32")["what"]
    assert F.plan("darwin")["how"] == "download" and F.plan("darwin")["source"] == "conda-forge"
    have = {"apt-get", "pkexec"}
    p = F.plan("linux", which=lambda n: f"/usr/bin/{n}" if n in have else None)
    if not (hasattr(os, "geteuid") and os.geteuid() == 0):
        assert p["how"] == "package" and p["argv"][:3] == ["pkexec", "sh", "-c"] and "libmpv2" in p["argv"][3]
    have = {"pacman"}
    p = F.plan("linux", which=lambda n: f"/usr/bin/{n}" if n in have else None)
    assert "pacman" in p.get("command", "") + " ".join(p.get("argv", []))
    assert F.plan("linux", which=lambda n: None)["how"] is None
    assert F.conda_subdir("darwin", "aarch64") == "osx-arm64" and F.conda_subdir("darwin", "x86_64") == "osx-64"


def test_release_page_fallback_parser():
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    html = ('<a href="/zhongfly/mpv-winbuild/releases/download/2026-08-21-abc/mpv-dev-lgpl-x86_64-20260821-git-abc.7z"'
            ' rel="nofollow">x</a><a href="/zhongfly/mpv-winbuild/releases/download/2026-08-21-abc/'
            'mpv-x86_64-20260821-git-abc.7z">y</a><a href="/other/repo/releases/download/t/evil.7z">z</a>')
    rel = F.parse_release_page(html, "zhongfly/mpv-winbuild", "2026-08-21-abc")
    assert [a["name"] for a in rel["assets"]] == ["mpv-dev-lgpl-x86_64-20260821-git-abc.7z",
                                                  "mpv-x86_64-20260821-git-abc.7z"]
    a = F.choose_windows_asset(rel, "x86_64")
    assert a["browser_download_url"].startswith("https://github.com/zhongfly/mpv-winbuild/releases/download/")


@pytest.fixture
def http_dir(tmp_path):
    import functools
    import http.server
    import threading
    root = tmp_path / "www"
    root.mkdir()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(root)))
    srv.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_download_checksum_progress_and_cancel(http_dir, tmp_path, monkeypatch):
    import hashlib
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    root, base = http_dir
    data = os.urandom(3_000_000)
    (root / "f.bin").write_bytes(data)
    seen = []
    out = F.download(f"{base}/f.bin", tmp_path / "d" / "f.bin", lambda f, t: seen.append(f), label="x",
                     sha256=hashlib.sha256(data).hexdigest())
    assert out.read_bytes() == data and not (tmp_path / "d" / "f.bin.part").exists()
    with pytest.raises(RuntimeError, match="checksum"):
        F.download(f"{base}/f.bin", tmp_path / "e" / "f.bin", sha256="0" * 64)
    assert not (tmp_path / "e" / "f.bin").exists()
    with pytest.raises(F.Cancelled):
        F.download(f"{base}/f.bin", tmp_path / "c" / "f.bin", cancel=lambda: True)


def test_windows_fetch_end_to_end_with_a_fake_release(tmp_path, monkeypatch):
    py7zr = pytest.importorskip("py7zr")
    import hashlib
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    src = tmp_path / "src"
    (src / "mpv-dev").mkdir(parents=True)
    (src / "mpv-dev" / "libmpv-2.dll").write_bytes(b"MZ fake dll")
    (src / "mpv-dev" / "readme.txt").write_text("x")
    arc = tmp_path / "mpv-dev-lgpl-x86_64-20260821-git-abc.7z"
    with py7zr.SevenZipFile(arc, "w") as z:
        z.writeall(src / "mpv-dev", "mpv-dev")
    digest = hashlib.sha256(arc.read_bytes()).hexdigest()
    rel = {"tag_name": "t", "assets": [{"name": arc.name, "browser_download_url": "u", "digest": f"sha256:{digest}"}]}
    monkeypatch.setattr(F, "windows_release", lambda token=None: rel)
    monkeypatch.setattr(F, "machine", lambda: "x86_64")
    got = []

    def fake_download(url, dst, progress=None, cancel=None, label="", sha256=None):
        got.append(sha256)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arc, dst)
        return dst
    monkeypatch.setattr(F, "download", fake_download)
    dest = tmp_path / "user"
    dest.mkdir()
    dll = F.fetch_windows(dest)
    assert dll == dest / "libmpv-2.dll" and dll.read_bytes() == b"MZ fake dll"
    assert got == [digest] and "LGPL" in (dest / "SOURCE.txt").read_text()
    assert not (dest / ".download").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shell-script stand-in for micromamba")
def test_conda_fetch_installs_prunes_and_swaps_in(tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    fake = tmp_path / "mm" / "micromamba"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\n# micromamba create -y -p PREFIX ...\nwhile [ "$1" != "-p" ]; do shift; done\n'
                    'p="$2"\nmkdir -p "$p/lib/python3.14" "$p/bin" "$p/include" "$p/share/doc" "$p/lib/pkgconfig"\n'
                    'echo lib > "$p/lib/libmpv.2.dylib"; echo lib > "$p/lib/libmpv.so.2"; echo a > "$p/lib/libx.a"\n'
                    'echo py > "$p/lib/python3.14/os.py"; echo exe > "$p/bin/yt-dlp"\necho "Transaction finished"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(F, "ensure_micromamba", lambda dest, subdir, progress=None, cancel=None: fake)
    dest = tmp_path / "user"
    lines = []
    lib = F.fetch_conda(dest, lambda f, t: lines.append(t), subdir="osx-arm64")
    assert lib == dest / "env" / "lib" / "libmpv.2.dylib" and lib.is_file()
    env = dest / "env"
    assert not (env / "bin").exists() and not (env / "include").exists() and not (env / "lib" / "python3.14").exists()
    assert not (env / "lib" / "libx.a").exists() and not (dest / "env.new").exists()
    assert any("Transaction finished" in t for t in lines)
    lib2 = F.fetch_conda(dest, subdir="linux-64")                       # a second run replaces the env
    assert lib2 == dest / "env" / "lib" / "libmpv.so.2"


def test_failed_micromamba_reports_its_output(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("shell script")
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    fake = tmp_path / "micromamba"
    fake.write_text("#!/bin/sh\necho 'critical libmamba Could not solve for environment specs'\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(F, "ensure_micromamba", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="Could not solve"):
        F.fetch_conda(tmp_path / "u", subdir="osx-64")
    assert not (tmp_path / "u" / "env").exists()


def test_fetch_job_reports_on_the_ui_thread(qapp, qtbot, tmp_path):
    import threading
    from AIO_QC.tools.frame_extractor.widgets.libmpv_dialog import MpvFetchJob
    main = threading.get_ident()
    seen = []

    def fake(dest, progress, cancel):
        progress(0.5, "half")
        return ""
    job = MpvFetchJob(str(tmp_path), None, fetch_fn=fake)
    job.progress.connect(lambda f, t: seen.append(("p", f, t, threading.get_ident() == main)))
    job.succeeded.connect(lambda p: seen.append(("ok", p, threading.get_ident() == main)))
    job.start()
    qtbot.waitUntil(lambda: not job.running, timeout=5000)
    assert seen == [("p", 0.5, "half", True), ("ok", "", True)]

    def slow(dest, progress, cancel):
        import time
        while not cancel():
            time.sleep(0.02)
        from AIO_QC.tools.frame_extractor.core.mpv_fetch import Cancelled
        raise Cancelled()
    job2 = MpvFetchJob(str(tmp_path), None, fetch_fn=slow)
    fails = []
    job2.failed.connect(fails.append)
    job2.start()
    job2.cancel()
    qtbot.waitUntil(lambda: not job2.running, timeout=5000)
    assert fails == ["cancelled"]


def _fresh_fx(qapp, qtbot, tmp_path):
    from AIO_QC.tools.frame_extractor.app import MainWindow
    w = MainWindow()
    w.output_dir = str(tmp_path)
    qtbot.addWidget(w)
    return w


def test_missing_libmpv_is_downloaded_in_the_background(qapp, qtbot, tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.core import mpv_fetch, mpv_lib
    monkeypatch.delenv("AIOQC_NO_LIBMPV_DOWNLOAD", raising=False)
    monkeypatch.setattr(mpv_fetch, "plan", lambda *a, **k: {"how": "download", "auto": True, "what": "x",
                                                             "source": "test"})
    calls = []
    monkeypatch.setattr(mpv_fetch, "fetch", lambda dest, progress, cancel: (calls.append(dest), "")[1])
    w = _fresh_fx(qapp, qtbot, tmp_path)
    w.settings.remove("playback/libmpv_fetch_failed")
    w.settings.setValue("playback/libmpv_auto", True)
    assert w._maybe_auto_fetch() is True
    assert w._maybe_auto_fetch() is False                    # once per session
    qtbot.waitUntil(lambda: bool(calls) and not w.libmpv_job(start=False), timeout=8000)
    assert calls == [mpv_lib.user_lib_dir()]
    w.close()


def test_failed_download_waits_before_trying_again(qapp, qtbot, tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.core import mpv_fetch
    monkeypatch.delenv("AIOQC_NO_LIBMPV_DOWNLOAD", raising=False)
    monkeypatch.setattr(mpv_fetch, "plan", lambda *a, **k: {"how": "download", "auto": True, "what": "x",
                                                             "source": "test"})

    def boom(dest, progress, cancel):
        raise RuntimeError("offline")
    monkeypatch.setattr(mpv_fetch, "fetch", boom)
    w = _fresh_fx(qapp, qtbot, tmp_path)
    w.settings.remove("playback/libmpv_fetch_failed")
    w.settings.setValue("playback/libmpv_auto", True)
    job = w.libmpv_job(start=True)
    qtbot.waitUntil(lambda: not job.running, timeout=5000)
    assert float(w.settings.value("playback/libmpv_fetch_failed")) > 0
    w2 = _fresh_fx(qapp, qtbot, tmp_path)                    # "next start": too soon to try again
    assert w2._maybe_auto_fetch() is False
    w.settings.setValue("playback/libmpv_auto", False)
    w.settings.remove("playback/libmpv_fetch_failed")
    w3 = _fresh_fx(qapp, qtbot, tmp_path)
    assert w3._maybe_auto_fetch() is False                   # turned off in the dialog
    w.settings.setValue("playback/libmpv_auto", True)
    for x in (w, w2, w3):
        x.close()


def test_linux_asks_once_before_using_the_package_manager(qapp, qtbot, tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.core import mpv_fetch
    monkeypatch.delenv("AIOQC_NO_LIBMPV_DOWNLOAD", raising=False)
    monkeypatch.setattr(mpv_fetch, "plan", lambda *a, **k: {
        "how": "package", "auto": False, "what": "x", "source": "package manager",
        "argv": ["pkexec", "sh", "-c", "apt-get install -y libmpv2"], "command": "sudo apt-get install libmpv2"})
    started = []
    monkeypatch.setattr(mpv_fetch, "fetch", lambda dest, progress, cancel: (started.append(1), "")[1])
    w = _fresh_fx(qapp, qtbot, tmp_path)
    w.settings.remove("playback/libmpv_asked")
    w.settings.remove("playback/libmpv_fetch_failed")
    assert w._maybe_auto_fetch() is True
    qtbot.waitUntil(lambda: bool(started), timeout=5000)     # the test harness answers the question "Yes"
    w2 = _fresh_fx(qapp, qtbot, tmp_path)
    assert w2._maybe_auto_fetch() is False                   # asked once, never again
    for x in (w, w2):
        x.close()


def test_dialog_get_button_runs_the_apps_job(qapp, qtbot, tmp_path, monkeypatch):
    from AIO_QC.tools.frame_extractor.core import mpv_fetch
    from AIO_QC.tools.frame_extractor.widgets.libmpv_dialog import LibmpvDialog
    monkeypatch.setattr(mpv_fetch, "plan", lambda *a, **k: {"how": "download", "auto": True, "what": "x",
                                                             "source": "test"})
    monkeypatch.setattr(mpv_fetch, "fetch", lambda dest, progress, cancel: (progress(0.3, "30 MB"), "")[1])
    w = _fresh_fx(qapp, qtbot, tmp_path)
    dlg = LibmpvDialog(w, w.settings, w.mpv_state, None, w.libmpv_job)
    qtbot.addWidget(dlg)
    assert dlg.get_btn.isEnabled()
    dlg.get_btn.click()
    qtbot.waitUntil(lambda: "installed" in dlg.fetch_msg.text(), timeout=5000)
    assert dlg.get_btn.isEnabled() and not dlg.cancel_btn.isVisible()
    dlg.auto.setChecked(False)
    assert w.settings.value("playback/libmpv_auto", True, type=bool) is False
    dlg.auto.setChecked(True)
    w.close()


@pytest.mark.skipif(not os.environ.get("AIOQC_TEST_CONDA_FETCH"),
                    reason="set AIOQC_TEST_CONDA_FETCH=1 to download mpv from conda-forge (~100 MB, ~1 min)")
def test_real_conda_fetch(tmp_path):
    """The macOS route for real (with the Linux packages - same micromamba / conda-forge mechanics): micromamba is
    fetched, mpv installed and pruned, and the libmpv loads and opens a video outside the app."""
    from AIO_QC.tools.frame_extractor.core import mpv_fetch as F
    lib = F.fetch_conda(tmp_path / "u", subdir=F.conda_subdir("linux"))
    r = subprocess.run([sys.executable, "-c", (
        "import sys,ctypes as C,time;sys.path.insert(0,%r)\n"
        "from AIO_QC.tools.frame_extractor.core import mpv_lib\n"
        "lib,msg=mpv_lib.probe(%r);h=lib.mpv_create()\n"
        "[lib.mpv_set_option_string(h,k,v) for k,v in ((b'vo',b'null'),(b'ao',b'null'))];lib.mpv_initialize(h)\n"
        "lib.mpv_command(h,(C.c_char_p*3)(b'loadfile',%r,None));t=time.time()\n"
        "while time.time()-t<15:\n e=lib.mpv_wait_event(h,1.0).contents\n"
        " if e.event_id in (7,8): print('EVENT',e.event_id);break\n") % (str(ROOT), str(lib),
                                                                       os.path.join(D, "a.mp4").encode())],
        capture_output=True, text=True, timeout=60)
    assert "EVENT 8" in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / "u" / "env" / "bin").exists()
