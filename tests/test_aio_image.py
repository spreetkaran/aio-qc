# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Image QC workspace on real Qt: open each dataset layout, run every QC tool, edit, zoom, sketch, switch modes."""
from __future__ import annotations

from pathlib import Path

import pytest

from . import fixtures as FX


@pytest.fixture
def shell(qapp, qtbot):
    from AIO_QC.shell import AppShell
    sh = AppShell(qapp)
    qapp.shell = sh
    qapp.main_window = sh
    w = sh.show("image")
    qtbot.addWidget(w)
    yield sh
    sh._closing = True
    if sh.studio is not None:
        sh.studio.shutdown()
        sh.studio.close()
    if sh.image is not None:
        sh.image._quit_reason = "test"
        sh.image.close()
    qapp.shell = None


def _open(sh, qtbot, root):
    w = sh.window("image")
    w.open_dataset(Path(root))
    assert w.session is not None, "dataset did not open"
    qtbot.waitUntil(lambda: w.session.index_complete, timeout=30000)
    return w


DATASETS = {"labelme": FX.labelme, "yolo": FX.yolo, "cls": FX.classification}


@pytest.mark.parametrize("kind", list(DATASETS))
def test_every_qc_tool_scans(shell, qtbot, tmp_path, kind):
    from AIO_QC.app import QC_TOOLS
    w = _open(shell, qtbot, DATASETS[kind](tmp_path / kind))
    opened = []
    for t in QC_TOOLS:
        if t.dialog is None or t.key == "fix_editor":
            continue
        if t.dialog.SUPPORTED is not None and w.session.type not in t.dialog.SUPPORTED:
            continue
        w.open_tool(t.key)
        d = w.open_dialogs.get(t.key)
        if d is None:
            continue
        if hasattr(d, "start_scan") and hasattr(d, "worker"):
            d.start_scan()
            qtbot.waitUntil(lambda: d.worker is None, timeout=60000)
        opened.append((t.key, len(getattr(d, "rows", []) or [])))
        d.close()
        qtbot.wait(5)
    from .conftest import MESSAGES
    print(kind, "tools:", opened, "| messages:", [m for m in MESSAGES if m[0] != "dialog"][:12])
    assert len(opened) >= 10


def test_navigation_zoom_tags_sketch(shell, qtbot, tmp_path):
    w = _open(shell, qtbot, FX.labelme(tmp_path / "lm"))
    for _ in range(3):
        w.a.next.trigger()
        qtbot.wait(20)
    c = w.canvas
    from PyQt6.QtCore import QPoint
    for pos in (c.viewport().rect().center(), QPoint(5, 5), QPoint(c.viewport().width() - 5, 10)):
        c.toggle_zoom(pos)
        qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
        c.layout_tags()
        c.toggle_zoom(pos)
        qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    c.zoom_by(1.5, c.viewport().rect().center())
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    c.fit_window()
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    w.sketch.toggle()
    assert w.sketch.active
    for k in ("pen", "line", "arrow", "rect", "ellipse", "circle", "erase", "pick"):
        w.sketch.set_tool(k)
    w.sketch.toggle()
    w.roi.toggle()
    w.roi.toggle()


def test_mode_switching_repeatedly(shell, qtbot, tmp_path):
    _open(shell, qtbot, FX.labelme(tmp_path / "lm"))
    for _ in range(4):
        shell.switch("data")
        qtbot.wait(30)
        shell.switch("image")
        qtbot.wait(30)
    shell.annotations_to_studio()
    st = shell.studio
    qtbot.waitUntil(lambda: st.current is not None and st.df is not None, timeout=20000)
    assert len(st.df) > 0


def test_data_studio_recipe(shell, qtbot, tmp_path):
    shell.switch("data")
    st = shell.studio
    p = FX.table(tmp_path / "t.csv")
    st.open_data(str(p))
    qtbot.waitUntil(lambda: st.df is not None, timeout=20000)
    st.add_step("text", {"cols": ["city"], "trim": True, "spaces": True, "case": "keep"}, ask=False)
    st.add_step("cast", {"cols": ["qty"], "type": "number"}, ask=False)
    st.add_step("derive", {"name": "total", "formula": "amount * qty", "text_cols": []}, ask=False)
    st.add_step("group", {"by": ["city"], "aggs": [["total", "sum"], ["id", "rows"]]}, ask=False)
    qtbot.waitUntil(lambda: st.df is not None and "total_sum" in st.df.columns, timeout=20000)
    for i in range(4):
        st.tabs.setCurrentIndex(i)
        qtbot.wait(50)
    st.undo()
    st.redo()
    qtbot.waitUntil(lambda: "total_sum" in st.df.columns, timeout=20000)


SKIP_ACTIONS = {"exit", "reset", "open_dir", "to_studio", "to_video", "next_mode", "ann_table", "watch"}


@pytest.mark.parametrize("kind", ["labelme", "yolo", "cls"])
def test_every_action(shell, qtbot, tmp_path, kind):
    """Trigger every menu action of the image window (dialogs answered automatically) — nothing may raise."""
    w = _open(shell, qtbot, DATASETS[kind](tmp_path / kind))
    w.canvas.select_shape(0) if w.canvas.shape_groups else None
    toggles = {a: a.isChecked() for a in vars(w.a).values() if hasattr(a, "isCheckable") and a.isCheckable()}
    for name, act in list(vars(w.a).items()):
        if name in SKIP_ACTIONS or not hasattr(act, "trigger"):
            continue
        if not act.isEnabled():
            continue
        act.trigger()
        qtbot.wait(15)
        for d in list(w.open_dialogs.values()):
            d.close()
        for top in list(qtbot_top_levels(w)):
            top.close()
    for act, was in toggles.items():                 # stored view options must not leak into later tests
        if act.isChecked() != was:
            act.trigger()
    from .conftest import MESSAGES
    print(kind, "messages:", [m for m in MESSAGES if m[0] != "dialog"][:15])


def qtbot_top_levels(main):
    from PyQt6.QtWidgets import QApplication, QDialog
    return [t for t in QApplication.topLevelWidgets() if isinstance(t, QDialog) and t.isVisible()]


@pytest.mark.parametrize("kind", ["labelme", "yolo"])
def test_inline_editing_with_mouse_and_keys(shell, qtbot, tmp_path, kind):
    import json
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    w = _open(shell, qtbot, DATASETS[kind](tmp_path / kind))
    w.resize(1200, 800)
    qtbot.wait(50)
    c = w.canvas
    assert c.shape_groups
    w.a.select.trigger()
    g = c.shape_groups[0]
    c.select_shape(g.index)
    qtbot.wait(20)
    before = json.dumps(w.session.load(w.current_key).shapes, default=str)
    # drag the first handle by 15 px
    assert c._handles, "no handles on the selected shape"
    h = c._handles[0]
    p = c.mapFromScene(h.sceneBoundingRect().center())
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
    for k in range(1, 6):
        QTest.mouseMove(c.viewport(), p + QPoint(3 * k, 2 * k))
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p + QPoint(15, 10))
    qtbot.wait(50)
    after = json.dumps(w.session.load(w.current_key).shapes, default=str)
    assert after != before, "dragging a handle did not change the file"
    # nudge with arrows, undo / redo
    c.setFocus()
    for key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
        QTest.keyClick(c, key)
    qtbot.wait(500)
    w.a.undo_edit.trigger()
    qtbot.wait(50)
    w.a.redo_edit.trigger()
    qtbot.wait(50)
    # wheel zoom, double-click, Space-pan
    QTest.mouseDClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(30, 30))
    qtbot.waitUntil(lambda: not c.is_animating(), timeout=3000)
    # delete the selected shape with Del, then undo
    n = len(w.session.load(w.current_key).shapes)
    c.select_shape(c.shape_groups[-1].index)
    QTest.keyClick(c, Qt.Key.Key_Delete)
    qtbot.wait(50)
    assert len(w.session.load(w.current_key).shapes) == n - 1
    w.a.undo_edit.trigger()
    qtbot.wait(50)
    assert len(w.session.load(w.current_key).shapes) == n


@pytest.mark.parametrize("kind", ["labelme", "yolo"])
def test_export_and_report_and_handoff(shell, qtbot, tmp_path, kind, monkeypatch):
    from PyQt6.QtGui import QDesktopServices
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda *a: True))
    w = _open(shell, qtbot, DATASETS[kind](tmp_path / kind))
    w.open_tool("export")
    d = w.open_dialogs["export"]
    for i in range(d.fmt.count()):
        d.fmt.setCurrentIndex(i)
        out = tmp_path / f"exp_{kind}_{d.fmt.currentData()}"
        d.out.setText(str(out))
        d.plan = None
        d.do_export()
        qtbot.waitUntil(lambda: d.worker is not None and not d.worker.isRunning() or "Exported" in d.status.text()
                        or "⚠" in d.status.text(), timeout=60000)
        qtbot.waitUntil(lambda: "Exported" in d.status.text() or "⚠" in d.status.text(), timeout=60000)
        assert "Exported" in d.status.text(), d.status.text()
    d.close()
    w.open_tool("report")
    r = w.open_dialogs["report"]
    for fmt in ("html", "pdf"):
        r.fmt.setCurrentIndex(r.fmt.findData(fmt))
        r.path.setText(str(tmp_path / f"report.{fmt}"))
        r._generate()
        qtbot.waitUntil(lambda: r.worker is None, timeout=60000)
        assert (tmp_path / f"report.{fmt}").exists(), r.status.text()
    r.close()


def test_data_prep_pages(shell, qtbot, tmp_path):
    import shutil
    from .fixtures import labelme
    src = labelme(tmp_path / "raw" / "a")
    shutil.copytree(src, tmp_path / "raw" / "b")
    w = shell.window("image")
    w.open_data_prep()
    dp = w.open_dialogs["data_prep"]
    pages = {p.TITLE: p for p in dp.pages}
    def run(page, setup):
        dp.nav.setCurrentRow(dp.pages.index(page))
        setup(page)
        dp._preview()
        qtbot.waitUntil(lambda: dp.job is None, timeout=60000)
        if dp.plan is not None and dp.plan.actions:
            dp._run()
            qtbot.waitUntil(lambda: dp.job is None, timeout=60000)
        return dp.status.text()
    out = {}
    out["integrity"] = run(pages["Integrity check"], lambda p: p.root.setText(str(tmp_path / "raw")))
    out["dupes"] = run(pages["Exact duplicates"], lambda p: p.root.setText(str(tmp_path / "raw")))
    out["orphans"] = run(pages["Find orphans"], lambda p: p.root.setText(str(tmp_path / "raw")))
    out["resize"] = run(pages["Resize images"], lambda p: (p.src.setText(str(tmp_path / "raw")),
                                                          p.dst.setText(str(tmp_path / "small")),
                                                          p.limit.setValue(100)))
    out["exif"] = run(pages["Fix EXIF rotation"], lambda p: p.root.setText(str(tmp_path / "raw")))
    out["pick"] = run(pages["Pick images to label"], lambda p: (p.src.setText(str(tmp_path / "raw")),
                                                               p.out.setText(str(tmp_path / "picked")),
                                                               p.n.setValue(3), p.skip.setChecked(False),
                                                               p.unlike.setChecked(False)))
    print(out)
    assert not any(v.startswith("⚠") for v in out.values()), out
    dp.close()
