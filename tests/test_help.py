# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""One Help Center for the whole app: every chapter renders, F1 in each mode opens its chapter, the shortcut page
covers every mode, and the keys that mean different things in different modes are listed."""
from __future__ import annotations

from .test_modes import shell3  # noqa: F401


def test_one_help_center(shell3, qtbot):  # noqa: F811
    from AIO_QC.widgets.help_center_bridge import open_help, shortcut_clashes
    sh = shell3
    for m in ("image", "video", "data"):
        sh.switch(m) if sh.mode else sh.show(m)
    hc = open_help(None, "image")
    qtbot.addWidget(hc)
    names = [c[0] for c in hc.chapters]
    titles = [c[1] for c in hc.chapters]
    assert names[:3] == ["aio-image-qc", "aio-data-studio", "aio-modes"]
    assert any(t.startswith("Frame Extractor ·") for t in titles)
    assert sum(t.startswith("Testing ·") for t in titles) == 2
    for n in names + ["shortcuts"]:
        hc.show_page(n)
        text = hc.view.toPlainText()
        assert len(text) > 200, n
        assert "FrameExtractor" not in text.replace("standalone FrameExtractor", "") and "Model Lab" not in text.replace("formerly the Frame Extractor's Model Lab", "") \
            .replace("formerly the Model Lab", ""), n
    hc.show_page("shortcuts")
    sc = hc.view.toPlainText()
    for mode in ("Image QC", "Frame Extractor", "Data Studio"):
        assert mode in sc, mode
    # F1 in each mode -> the same window, at that mode's chapter
    assert open_help(sh.image, "image") is hc and hc._hist[hc._pos] == "aio-image-qc"
    sh.studio.show_help()
    assert hc._hist[hc._pos] == "aio-data-studio"
    sh.video.open_help()
    assert hc._hist[hc._pos] == "01-welcome.html"
    # icons inside AIO QC's chapter come from the app
    hc.show_page("aio-image-qc")
    assert hc.icon_loader("tool-testing", 1.0) is not None
    # in-page links and app actions
    from PyQt6.QtCore import QUrl
    hc._link(QUrl("#models"))
    hc._link(QUrl("action:toolmap"))
    clashes = shortcut_clashes()
    assert "Ctrl+O" in clashes and len(clashes["Ctrl+O"]) >= 2
    hc.show_page("aio-modes")
    assert "Same keys, different modes" in hc.view.toPlainText()
    hc.close()
