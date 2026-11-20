# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Auto-sketched Flagged Screenshots: choose shapes per file (click on the picture or tick), the screenshot points at
them with the batch's marks, hides the other shapes, keeps files without choices as usual; choices persist."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage

from AIO_QC.core.autosketch import ChoiceStore
from AIO_QC.widgets.autosketch_dialog import AutoSketchDialog
from AIO_QC.widgets.batch_shots import BatchShots

from .test_batch_shots import _flag, _run, _win
from .test_aio_image import shell  # noqa: F401


def _near(c: QColor, hexc: str, tol=60) -> bool:
    q = QColor(hexc)
    return abs(c.red() - q.red()) + abs(c.green() - q.green()) + abs(c.blue() - q.blue()) < tol


def _count(img: QImage, hexc: str) -> int:
    n = 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            if _near(img.pixelColor(x, y), hexc):
                n += 1
    return n


def test_choose_by_click_and_save(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    keys = [s.key for s in w.session.samples]
    flagged = keys[1:4]
    _flag(w, flagged)
    w.settings.setValue("screenshot/auto_color", "#af52de")          # a colour the fixture images never have
    dlg = AutoSketchDialog(w, flagged)
    dlg.show()
    qtbot.waitUntil(lambda: dlg.preview.W > 0, timeout=5000)
    assert dlg.files.count() == 3 and dlg.shapes.count() == 3          # car, plate, wheel
    # click the car on the picture
    p = dlg.preview
    rect, k = p._frame()
    car = p.boxes[0]
    pos = QPointF(rect.x() + (car[0] + 4) * k, rect.y() + (car[1] + 4) * k)
    from PyQt6.QtTest import QTest
    QTest.mouseClick(p, Qt.MouseButton.LeftButton, pos=pos.toPoint())
    assert dlg.shapes.item(0).checkState() == Qt.CheckState.Checked and p.chosen == {0}
    assert any(e["t"] == "arrow" for e in p._els)                      # the marks preview live
    dlg._step(1)                                                       # D: next file
    qtbot.waitUntil(lambda: dlg.cur == flagged[1] and dlg.preview.W > 0, timeout=5000)
    dlg.shapes.item(2).setCheckState(Qt.CheckState.Checked)            # tick the wheel
    dlg.accept()
    st = ChoiceStore(w.session.root)                                   # saved on disk
    assert st.count(flagged[0]) == 1 and st.count(flagged[1]) == 1 and st.count(flagged[2]) == 0
    auto = {"auto": True, "style": "arrow_ring", "color": "#af52de", "text": "", "text_size": "M"}
    job = _run(qtbot, BatchShots(w, flagged, "spot", caption=False, sketch=False, existing="keep", auto=auto).start())
    assert job.written == 3 and job.auto_files == 2 and not job.errors
    shots = [QImage(str(tmp_path / "shots" / "ds" / f"{Path(k).stem}.png")) for k in flagged]
    ann = w.session.adapter.load(w.session.by_key[flagged[0]])
    W = int(ann.raw.get("imageWidth", 320)) if hasattr(ann, "raw") and isinstance(ann.raw, dict) else 320
    assert shots[0].width() >= W - 1                                   # auto-sketched: the whole image
    assert _count(shots[0], "#af52de") > 30 and _count(shots[1], "#af52de") > 30     # the marks
    assert _count(shots[2], "#af52de") == 0                            # no choice: saved as usual


def test_other_shapes_hidden(shell, qtbot, tmp_path):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    key = [s.key for s in w.session.samples][2]
    ann = w.session.adapter.load(w.session.by_key[key])
    st = ChoiceStore(w.session.root)
    st.set(key, ann.shapes, [2])                                       # only the wheel
    st.flush()
    w.session._autosketch = None
    from AIO_QC.widgets.batch_shots import ShotRenderer
    from AIO_QC.widgets.workers import load_qimage
    r = ShotRenderer(w, "whole")
    img = load_qimage(ann.image_path)
    try:
        r.render(key, img, ann, False, {"chosen": [2], "style": "ring", "color": "#af52de", "text": ""})
        assert [g.shape_data.get("label") for g in r.c.shape_groups] == ["wheel"]
    finally:
        r.close()


def test_dialog_options_and_choose(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _win(shell, qtbot, tmp_path)
    keys = [s.key for s in w.session.samples]
    _flag(w, keys[:2])
    d = w.open_batch_shots()
    assert d.auto is not None and not d.auto.isChecked()
    picked = {}

    def fake_exec(self):
        picked["keys"] = list(self.keys)
        self.store.set(self.keys[0], list(w.session.adapter.load(w.session.by_key[self.keys[0]]).shapes), [0])
        return 1
    monkeypatch.setattr(AutoSketchDialog, "exec", fake_exec)
    d.choose_shapes()
    assert picked["keys"] == keys[:2] and d.auto.isChecked()          # choosing shapes turns auto-sketch on
    assert "1 file(s) with chosen shapes" in d.note.text()
    job = _run(qtbot, d.start())
    assert job.auto_files == 1 and job.written == 2
