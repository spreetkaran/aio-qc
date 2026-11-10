# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""More real-Qt coverage: hand-off round trip, History revert, Data Studio exports / charts / closing with a save."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog

from . import fixtures as FX
from .test_aio_image import DATASETS, _open, shell  # noqa: F401  (fixture)


def test_handoff_round_trip(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.core.state import REVIEW_FLAG
    from AIO_QC.widgets.handoff_dialog import ExportBatchDialog, ImportBatchDialog
    w = _open(shell, qtbot, DATASETS["labelme"](tmp_path / "ds"))
    s = w.session
    keys = list(s.by_key)[:3]
    s.set_review(keys, REVIEW_FLAG, "test")
    ex = ExportBatchDialog(w)
    qtbot.addWidget(ex)
    ex.folder.setText(str(tmp_path / "out"))
    ex.name.setText("b1")
    assert ex._keys(), "no flagged files offered"
    ex._export()
    qtbot.waitUntil(lambda: ex.worker is None, timeout=30000)
    assert "Exported" in ex.status.text(), ex.status.text()
    batch = tmp_path / "out" / "b1"
    assert (batch / "batch.json").is_file()
    # the annotator edits one returned file
    m = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    ann = batch / m["items"][0]["ann"]
    d = json.loads(ann.read_text(encoding="utf-8"))
    d["shapes"][0]["label"] = "truck"
    ann.write_text(json.dumps(d), encoding="utf-8")
    im = ImportBatchDialog(w)
    qtbot.addWidget(im)
    im.path.setText(str(batch))
    im._load()
    qtbot.waitUntil(lambda: im.worker is None and bool(im.rows), timeout=30000)
    assert im._ticked(), [r.state for r in im.rows]
    im._merge()
    qtbot.waitUntil(lambda: im.worker is None, timeout=30000)
    key = next(k for k in s.by_key if Path(s.by_key[k].ann_path).name == ann.name)
    assert s.load(key).shapes[0]["label"] == "truck"
    ex.close()
    im.close()


def test_history_revert(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, DATASETS["labelme"](tmp_path / "ds"))
    w.resize(1200, 800)
    key = w.current_key
    before = w.session.load(key).shapes[0]["label"]
    w.inline.on_relabel(0, 2)                                # 2nd class -> backup + edit
    qtbot.wait(50)
    assert w.session.load(key).shapes[0]["label"] != before
    w._show_history()
    hp = w.history_panel
    qtbot.waitUntil(lambda: hp.vers.topLevelItemCount() > 0, timeout=5000)
    hp.vers.setCurrentItem(hp.vers.topLevelItem(0))
    hp._preview()
    hp._revert()
    qtbot.wait(50)
    assert w.session.load(key).shapes[0]["label"] == before


@pytest.fixture
def studio(shell, qtbot, tmp_path):  # noqa: F811
    shell.switch("data")
    st = shell.studio
    st.open_data(str(FX.table(tmp_path / "t.csv")))
    qtbot.waitUntil(lambda: st.df is not None, timeout=20000)
    return st


def test_studio_exports(studio, qtbot, tmp_path, monkeypatch):
    st = studio
    for ext in (".csv", ".parquet", ".json", ".jsonl", ".xlsx"):
        p = tmp_path / f"out{ext}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, p=p, **k: (str(p), "")))
        st.export_data()
        qtbot.waitUntil(lambda p=p: p.is_file() and not st.jobs, timeout=20000)
    for engine in ("pandas", "spark"):
        outs = iter([(str(tmp_path / "r.parquet"), ""), (str(tmp_path / f"{engine}.py"), "")])
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: next(outs)))
        st.export_code(engine)
        code = (tmp_path / f"{engine}.py").read_text(encoding="utf-8")
        compile(code, f"{engine}.py", "exec")
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "p.html"), "")))
    st.export_report()
    qtbot.waitUntil(lambda: (tmp_path / "p.html").is_file() and not st.jobs, timeout=20000)


def test_studio_charts(studio, qtbot):
    from AIO_QC.widgets.studio.chart_view import TYPES
    st = studio
    st.tabs.setCurrentIndex(2)
    ch = st.chart
    drawn = 0
    for t in TYPES:
        ch.type.setCurrentText(t)
        for xi in range(ch.x.count()):
            ch.x.setCurrentIndex(xi)
            for yi in range(min(3, ch.y.count())):
                ch.y.setCurrentIndex(yi)
                err = (ch.canvas.spec or {}).get("error", "")
                assert not err.startswith("Cannot draw this"), (t, ch.x.currentText(), ch.y.currentText(), err)
                ch.canvas.grab()                        # paint it
                drawn += 1
    assert drawn > 10


def test_studio_close_asks_to_save(studio, qtbot, tmp_path, monkeypatch):
    st = studio
    st.add_step("cast", {"cols": ["qty"], "type": "number"}, ask=False)
    qtbot.waitUntil(lambda: not st.jobs, timeout=20000)
    assert st.ws.dirty
    p = tmp_path / "w.aioqcws"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(p), "")))
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save))
    assert st.maybe_save()                              # Save -> save as
    assert not st.ws.dirty
    assert any(f.name.startswith("w") for f in tmp_path.iterdir() if f.suffix != ".csv")
