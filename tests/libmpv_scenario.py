# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Real libmpv playback run (started by test_libmpv.test_real_libmpv_playback under xvfb-run).
Prints 'R <name> => <value>' lines and exits; uses a QTimer chain (no nested event loops)."""
import os
import sys

sys.path.insert(0, os.environ["AIOQC_ROOT"])
from AIO_QC import __main__ as M  # noqa: E402

M._process_setup_before_qt()
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog  # noqa: E402

app = QApplication(sys.argv)
M._process_setup_after_qt(app)
from AIO_QC.shell import AppShell  # noqa: E402

sh = AppShell(app)
app.shell = app.main_window = sh
sh.show("image")
MEDIA = os.environ["MEDIA"]
R, S = {}, {}


def v():
    return sh.video


def centre():
    c = v().canvas
    img = app.primaryScreen().grabWindow(0)
    p = c.mapToGlobal(c.rect().center())
    q = img.toImage().pixelColor(p.x(), p.y())
    return q.red() + q.green() + q.blue()


def chain(*steps):
    def nxt(i=0):
        if i >= len(steps):
            return
        try:
            ms = steps[i]()
        except Exception:
            import traceback
            traceback.print_exc()
            os._exit(3)
        QTimer.singleShot(ms or 0, lambda: nxt(i + 1))
    QTimer.singleShot(300, nxt)


def load():
    sh.switch("video")
    w = v()
    w.settings.setValue("playback/libmpv", os.environ["LIBMPV"])
    w.settings.setValue("playback/engine", "mpv")
    w.load_video(os.path.join(MEDIA, "a.mp4"))
    return 4000


def settle():
    """Wait (up to 20 s, slow machines / parallel load) for libmpv to open the file and its picture check."""
    import time
    S.setdefault("t0", time.monotonic())
    w, m = v(), v().mpv
    if (m is not None and m.loaded and getattr(w, "_mpv_verified", False)) or time.monotonic() - S["t0"] > 20:
        return 0
    QTimer.singleShot(250, settle_again)
    return None


def settle_again():
    if settle() == 0:
        chain_rest()


def loaded():
    for d in app.topLevelWidgets():
        if isinstance(d, QDialog) and d.isVisible():
            d.close()
    w, m = v(), v().mpv
    R["gpu"] = w.render_label.text()
    R["mpv loaded"] = bool(m and m.loaded and not m.failed)
    R["verified"] = bool(getattr(w, "_mpv_verified", False))
    w.goto_frame(50)
    return 800


def play():
    v().play()
    return 1500


def playing():
    w = v()
    R["mpv playing"] = bool(w._mpv_playing and w.canvas.mpv_showing)
    R["frame advanced"] = w.current_frame > 50
    R["mpv picture not black"] = centre() > 60
    w.pause()
    return 1000


def paused():
    w = v()
    R["paused exact"] = bool(w.current is not None and w.current.bgr is not None and w.current.idx == w.current_frame
                             and not w.canvas.mpv_showing)
    S["f"] = w.current_frame
    w.step_frames(1)
    return 600


def stepped():
    R["step +1"] = v().current_frame == S["f"] + 1
    v().play()
    return 400


def hide():
    sh.switch("image")
    return 400


def hidden():
    R["paused when hidden"] = not v().is_playing
    sh.switch("video")
    v().play()
    return 7000


def end():
    R["reached end"] = bool(v().ended)
    v().load_video(os.path.join(MEDIA, "b.mp4"))
    return 2500


def second():
    v().play()
    return 1000


def done():
    w = v()
    R["second video via mpv"] = bool(w._mpv_playing)
    w.pause()
    R["failure"] = w.mpv.failed if w.mpv else "no mpv"
    for k, val in R.items():
        print(f"R {k} => {val}", flush=True)
    os._exit(0)


def chain_rest():
    chain(loaded, play, playing, paused, stepped, hide, hidden, end, second, done)


def start():
    load()
    QTimer.singleShot(1000, settle_again)


QTimer.singleShot(300, start)
app.exec()
