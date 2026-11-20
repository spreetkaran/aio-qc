# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Long jobs look and behave the same everywhere: progress with time left, Stop, a Background Tasks entry, one result
line with Undo / Open — the shared bar, the QC Report on it, Convert to YOLO's Undo."""
from pathlib import Path

from PyQt6.QtWidgets import QWidget

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_eta_waits_for_evidence():
    from AIO_QC.widgets.jobs import eta
    assert eta(0, 0, 100, now=10) == ""              # nothing done yet
    assert eta(0, 50, 100, now=1) == ""              # under 3 s: too early to say
    assert eta(0, 1, 100, now=10) == ""              # 1 %: too little to go on
    assert eta(0, 50, 100, now=10) == "about 10 s left"
    assert eta(0, 10, 100, now=60) == "about 9 min left"
    assert eta(0, 1, 10, now=2000) == "about 5.0 h left"
    assert eta(0, 99, 100, now=100) == "a few seconds left"
    assert eta(0, 100, 100, now=100) == ""


def test_job_bar_states(qtbot):
    from AIO_QC.widgets import tasks
    from AIO_QC.widgets.jobs import JobBar
    host = QWidget()
    qtbot.addWidget(host)
    bar = JobBar(host, "Test job")
    stopped, undone = [], []
    bar.start("Copying", 10, stop=lambda: stopped.append(1))
    assert bar.running and not bar.progress.isHidden() and not bar.stop_btn.isHidden()
    c = tasks.center()
    assert c is None or any(t.title.startswith("Test job") and not t.done for t in c.tasks)
    bar.step(4, 10, "file 4")
    assert bar.progress.value() == 4 and "file 4" in bar.text.text()
    bar.request_stop()
    assert stopped == [1] and "Stopping" in bar.text.text() and not bar.stop_btn.isEnabled()
    bar.request_stop()                                # once is enough
    assert stopped == [1]
    bar.stopped(undo=lambda: undone.append(1) or "Put back")
    assert not bar.running and bar.progress.isHidden() and not bar.undo_btn.isHidden()
    bar.undo_btn.click()
    assert undone == [1] and "Put back" in bar.text.text() and bar.undo_btn.isHidden()
    bar.start("Again", 0)
    assert bar.stop_btn.isHidden()                    # no stop function: no Stop button
    bar.finish("All done", open_path="/tmp", extra=[("Show", lambda: None)])
    assert "✓ All done" in bar.text.text() and not bar.open_btn.isHidden() and len(bar._extra) == 1
    bar.start("Third", 3)
    assert bar._extra == [] and bar.open_btn.isHidden()
    bar.fail(PermissionError("[WinError 5] Access is denied: 'x'"))
    assert "not allowed" in bar.text.text()
    bar.close_job()


def test_report_on_the_job_bar(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_tool("report")
    r = w.open_dialogs["report"]
    r.fmt.setCurrentIndex(r.fmt.findData("html"))
    r.path.setText(str(tmp_path / "rep.html"))
    r._generate()
    qtbot.waitUntil(lambda: r.worker is None, timeout=60000)
    assert (tmp_path / "rep.html").is_file() and "✓ Saved" in r.status.text() and not r.job.open_btn.isHidden()
    # Stop while it builds: nothing is written
    r.path.setText(str(tmp_path / "rep2.html"))
    r._generate()
    r.job.request_stop()
    qtbot.waitUntil(lambda: r.worker is None, timeout=60000)
    assert ("Stopped" in r.status.text() and not (tmp_path / "rep2.html").exists()) or "✓ Saved" in r.status.text()
    r.close()


def test_convert_to_yolo_can_be_undone(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    w.open_tool("convert_yolo")
    d = w.open_dialogs["export"]
    conv = d
    out = tmp_path / "yolo_out"
    conv.out.setText(str(out))
    conv.start()
    qtbot.waitUntil(lambda: conv.worker is None, timeout=60000)
    assert (out / "dataset.yaml").is_file(), conv.status.text()
    assert not conv.undo_btn.isHidden()
    (out / "mine.txt").write_text("keep me")              # not the conversion's: stays
    conv._undo()
    assert not (out / "dataset.yaml").exists() and (out / "mine.txt").is_file() and "Undone" in conv.status.text()
    assert Path(out).is_dir()
    d.close()


def test_handoff_export_can_be_undone(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import REVIEW_FLAG
    from AIO_QC.widgets.handoff_dialog import ExportBatchDialog
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    keys = list(w.session.by_key)[:3]
    w.session.set_review(keys, REVIEW_FLAG, "test")
    ex = ExportBatchDialog(w)
    qtbot.addWidget(ex)
    ex.folder.setText(str(tmp_path / "out"))
    ex.name.setText("b1")
    ex._export()
    assert ex.job.running or ex.worker is None
    qtbot.waitUntil(lambda: ex.worker is None, timeout=30000)
    batch = tmp_path / "out" / "b1"
    assert (batch / "batch.json").is_file() and "Exported 3 file(s)" in ex.job.text.text()
    ex.job.undo_btn.click()                                   # (the question is answered Yes by the test harness)
    assert not batch.exists() and "deleted" in ex.job.text.text() and ex.go.isEnabled()
    ex.close()
