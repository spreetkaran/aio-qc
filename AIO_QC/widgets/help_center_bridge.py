# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC's one Help Center (F1 / Help in every mode): the Image QC guide, Data Studio, "Modes & working across
tools", the Frame Extractor's chapters and Testing's - one window per app with contents, search, back / forward,
zoom, a printable copy, and one keyboard-shortcut page for all three modes (with the keys that mean different things
in different modes listed).
"""
from __future__ import annotations

import re

_HC = {"w": None}

PAGES = {"image": "aio-image-qc", "data": "aio-data-studio", "modes": "aio-modes", "testing": "09-model-lab.html",
         "video": "01-welcome.html"}


def _shell():
    from PyQt6.QtWidgets import QApplication
    return getattr(QApplication.instance(), "shell", None)


# ------------------------------------------------------------------ chapters
def _strip(html):
    html = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
    m = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.S)
    return m.group(1) if m else html


def modes_html():
    from AIO_QC.shell import NAMES
    clash = shortcut_clashes()
    rows = "".join(f"<tr><td><span class='k'>{k}</span></td>" +
                   "".join(f"<td>{v.get(m, '—')}</td>" for m in ("image", "video", "data")) + "</tr>"
                   for k, v in sorted(clash.items()))
    clash_table = ("<table class='t'><tr><th>Keys</th>" + "".join(f"<th>{NAMES[m]}</th>" for m in
                                                                  ("image", "video", "data")) + "</tr>" + rows +
                   "</table>") if rows else "<p class='muted'>(Open each mode once to list its keys here.)</p>"
    return f"""
<h1>Modes &amp; working across tools</h1>
<p class="lead">AIO QC is one app with three modes, each a complete window of its own: the <b>Frame Extractor</b> (videos → frames
for annotation), <b>Image QC</b> (annotated image datasets) and <b>Data Studio</b> (tables). The app starts in Image QC.
<b>Testing</b> (measure and compare detection models) is part of Image QC's <i>Models</i> tab.</p>
<h2>Switching</h2>
<p>The switch in the top-right corner of the menu bar, <span class="k">Ctrl+Alt+M</span> (next mode) or
<i>File ▸ Switch to …</i>, in the order Frame Extractor → Image QC → Data Studio. A mode is loaded the first time you use it; switching keeps each window's state (the
Frame Extractor pauses when hidden). <code>aio-qc --mode image|video|data</code> starts in a mode;
<code>aio-qc video.mp4</code> opens a video, <code>aio-qc table.csv</code> a table. Closing any mode window quits the
app (each gets to save first; a running batch extraction is asked about).</p>
<h2>Between the tools</h2>
<table class="t">
<tr><th>From</th><th>To</th><th>How</th></tr>
<tr><td>Frame Extractor</td><td>Image QC</td><td><i>File ▸ Open output in Image QC</i>, the review gallery's
<i>Open in Image QC</i>, or the batch queue's button: the extracted frames with their pre-labels become the dataset
(YOLO pre-labels and unlabelled frames get LabelMe files added next to them after you confirm).</td></tr>
<tr><td>Image QC</td><td>Frame Extractor</td><td><i>File ▸ Show Source Video in Frame Extractor</i> (or right-click a
file): the video the image came from, at its frame. Frames saved by the Frame Extractor remember their video.</td></tr>
<tr><td>Testing</td><td>Frame Extractor</td><td><i>6 · Review videos ▸ Review in player</i>: the videos play one after
another with every model's detections; Testing only sends the playlist.</td></tr>
<tr><td>Testing</td><td>Image QC review</td><td><i>Send to review</i> on the Failures and Compare A / B tabs flags the
images and filters Image QC's file list to them.</td></tr>
</table>
<h2>One of everything</h2>
<p><b>Models:</b> weights added anywhere (Image QC ▸ Models &amp; Environment, the Frame Extractor's Settings ▸ Models,
Testing) are offered everywhere; one Python environment runs .pt / .engine models. Testing and Model Check share one
prediction cache per dataset (<code>.aio_qc/predictions.db</code>).<br>
<b>Background tasks:</b> one indicator in every mode's status bar lists long jobs (extraction, batch, timeline
analysis, exports, Testing, scans, Data Prep) — click it for the list and Cancel.<br>
<b>Settings:</b> every mode has its own Settings window - the gear next to the mode switch, File ▸ Settings…
(Ctrl+,) - searchable, applied at once, with reset and import / export; its General section is shared by all modes.<br>
<b>Animations:</b> one setting (Settings ▸ General ▸ Animations). <b>Theme:</b> the system's light / dark, everywhere.
<b>Log:</b> one log file (<code>aio_qc_tool.log</code>).</p>
<h2>Get started</h2>
<p><a href="action:getstarted-image">Image QC — the 1-minute tour</a> ·
<a href="action:getstarted-video">Frame Extractor — Get Started cards</a> ·
<a href="action:toolmap">Which Image QC tool for what?</a></p>
<h2>Same keys, different modes</h2>
<p>Each mode has its own window, so a key does what that window's menu says. These keys exist in more than one mode
with different commands (within one window every key is unique):</p>
{clash_table}
"""


def _chapters():
    from AIO_QC import __version__
    from AIO_QC.help_guide import guide_html
    from AIO_QC.widgets.studio.guide import guide_html as studio_html
    return [
        ("aio-image-qc", "Image QC — User Guide", "image qc dataset labelme yolo qc tools review models testing",
         _strip(guide_html(__version__))),
        ("aio-data-studio", "Data Studio — Guide", "data studio tables csv excel parquet pandas spark recipe chart",
         _strip(studio_html())),
        ("aio-modes", "Modes & working across tools", "modes switch frame extractor testing open in qc source video "
                                                       "review registry tasks shortcuts clashes", modes_html()),
    ]


# ------------------------------------------------------------------ shortcuts of every mode
def _window_entries(win, mode_name):
    from PyQt6.QtGui import QAction
    out, seen = [], set()
    for a in win.findChildren(QAction):
        keys = [k.toString() for k in a.shortcuts() if k.toString()]
        name = a.text().replace("&", "").replace("\n", " ").strip()
        if not keys or not name or (name, tuple(keys)) in seen:
            continue
        seen.add((name, tuple(keys)))
        out.append((mode_name, name, keys))
    return out


def all_shortcut_entries():
    """[(category, command, [keys])] of every mode window that exists (Frame Extractor: its configured keys)."""
    sh = _shell()
    out = []
    if sh is None:
        return out
    if sh.image is not None:
        out += _window_entries(sh.image, "Image QC")
    if sh.video is not None:
        try:
            out += [(f"Frame Extractor › {c}", n, k) for c, n, k in sh.video._shortcut_entries()]
        except Exception:
            out += _window_entries(sh.video, "Frame Extractor")
    if sh.studio is not None:
        out += _window_entries(sh.studio, "Data Studio")
    return out


def shortcut_clashes() -> dict:
    """{key: {mode: command}} for keys bound in more than one mode to different commands."""
    sh = _shell()
    per = {}
    if sh is None:
        return {}
    for mode, win in (("image", sh.image), ("video", sh.video), ("data", sh.studio)):
        if win is None:
            continue
        for _c, name, keys in _window_entries(win, mode):
            for k in keys:
                per.setdefault(k, {}).setdefault(mode, name)
    return {k: v for k, v in per.items() if len(v) > 1 and len({n.lower() for n in v.values()}) > 1}


# ------------------------------------------------------------------ the window
def _icon(name, dpr):
    from PyQt6.QtCore import QSize
    from AIO_QC.utils import newIcon
    return newIcon(name).pixmap(QSize(20, 20), dpr).toImage()


def _actions():
    def toolmap():
        sh = _shell()
        if sh is not None:
            sh.window("image").show_tool_map()

    def gs_image():
        sh = _shell()
        if sh is not None:
            sh.switch("image") if sh.mode != "image" else None
            sh.window("image").show_get_started()

    def gs_video():
        sh = _shell()
        if sh is not None:
            sh.switch("video") if sh.mode != "video" else None
            sh.window("video").show_get_started()

    def customize():
        sh = _shell()
        if sh is not None:
            sh.switch("video") if sh.mode != "video" else None
            sh.window("video").show_shortcuts()
    return {"toolmap": toolmap, "getstarted-image": gs_image, "getstarted-video": gs_video, "customize": customize}


def live_help():
    """The shared Help Center if it is open and not deleted (it is one window for every mode), else None."""
    from PyQt6 import sip
    w = _HC["w"]
    return None if w is None or sip.isdeleted(w) else w


def open_help(parent=None, page=None):
    from PyQt6 import sip
    from AIO_QC.tools.frame_extractor.widgets import help_center as HC
    w = _HC["w"]
    if w is None or sip.isdeleted(w):
        HC.EXTRA_CHAPTERS[:] = _chapters()
        try:
            from AIO_QC.tools.frame_extractor.app import MOUSE_HELP
        except Exception:
            MOUSE_HELP = []
        w = HC.HelpCenter(None, all_shortcut_entries, MOUSE_HELP, actions=_actions(), icon_loader=_icon)
        from AIO_QC.utils import newIcon
        w.setWindowIcon(newIcon("info"))
        _HC["w"] = w
    else:
        HC.EXTRA_CHAPTERS[:] = _chapters()                  # the modes chapter lists the windows built since
        w.chapters = HC.load_chapters()
        w.refresh()
    page = PAGES.get(page, page)
    if page or not w._hist:
        w.show_page(page or "aio-image-qc")
    w.show()
    w.raise_()
    w.activateWindow()
    return w
