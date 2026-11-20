# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Model Check end to end on the REAL inference server: a stand-in Python runs core/infer_server.py with a toy
Ultralytics (tests/infer_fake.py) — probe, load, batched predict, outlines, the prediction cache, then everything the
window does with the result: findings (missing / mismatch / missed / poor fit), live thresholds, every analysis
view and its drill-downs, an audit with verdicts, accept / undo, reject / un-reject, flag, the report card CSV,
Improve Model reading the same run, and SAM outlines through the same server."""
import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog

from . import fixtures as FX
from .infer_fake import fake_python, models_config
from .test_aio_image import _open, shell  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in Python is a shell script")


def _tick(d, pred):
    hits = []
    for t in range(d.table.rowCount()):
        r = d._row_at(t)
        on = r is not None and pred(r)
        d.table.item(t, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        if on:
            hits.append(r)
    return hits


@pytest.fixture
def run(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds"))
    models_config(w.session.root, tmp_path)
    w.inference.set_python(fake_python(tmp_path))
    w.open_tool("model_check")
    d = w.open_dialogs["model_check"]
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=120000)
    assert d.ctx.get("eval") is not None, (d.status.text(), d.errors[:3])
    yield w, d
    w.inference.stop()


def test_model_check_on_the_real_server(run, qtbot, tmp_path, monkeypatch):
    from AIO_QC.core.model_check import MISMATCH, MISSED, MISSING
    w, d = run
    ctx = d.ctx
    assert ctx["ran"] == 12 and ctx["cached"] == 0 and w.inference.ready
    kinds = {r.cells[0] for r in d.rows if r is not None}
    assert {MISSING, MISMATCH, MISSED} <= kinds, kinds
    assert any(r.cells[0] == MISSING and r.cells[2] == "car" for r in d.rows if r)       # the unlabelled car
    ev = ctx["eval"]
    assert set(ev.models) == {"md", "ms"} and len(ev.keys) == 12
    # outlines from the segmentation model reached the explorer
    qtbot.waitUntil(lambda: d.explorer.state is not None, timeout=20000)
    data = d.explorer.data
    assert any(data.p_poly(i) is not None for i in range(len(data.p_img)))
    # a second run: every prediction from the cache, nothing sent to the server
    d.start_scan()
    qtbot.waitUntil(lambda: d.worker is None, timeout=120000)
    assert d.ctx["ran"] == 0 and d.ctx["cached"] == 12
    n0 = sum(r is not None for r in d.rows)
    # live thresholds: stricter confidence, fewer findings — instantly, no new inference
    d.conf.setValue(0.85)
    qtbot.waitUntil(lambda: sum(r is not None for r in d.rows) != n0, timeout=20000)
    d.conf.setValue(0.25)
    qtbot.waitUntil(lambda: sum(r is not None for r in d.rows) == n0, timeout=20000)
    # every analysis view, and their drill-downs
    d.tabs.setCurrentWidget(d._analysis)
    d._layout_for_tab()
    d._refresh_analysis()
    for i in range(d.v_group.count()):
        if d.v_group.model().item(i).isEnabled():
            d.v_group.setCurrentIndex(i)
    d.v_group.setCurrentIndex(0)
    for fn in ("_refresh_confusion", "_refresh_errors", "_refresh_boxes", "_refresh_slices", "_refresh_current_inner",
               "_refresh_cards"):
        getattr(d, fn)()
    assert d._last_report and any(r["label"] == "car" for r in d._last_report)
    for r in range(min(3, d.report_tbl.rowCount() if hasattr(d, "report_tbl") else 0)):
        d._report_activated(r, 0)
    out = tmp_path / "card.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    d._export_report()
    assert out.is_file() and "car" in out.read_text(encoding="utf-8-sig")
    # Findings: cards and table
    d.tabs.setCurrentWidget(d._findings)
    d._layout_for_tab()
    d._set_findings_view(0)
    d._set_findings_view(1)
    d.show_kind(MISSING)
    assert d.filter_box.currentData() == MISSING
    d.filter_box.setCurrentIndex(0)
    # accept the missing cars, then undo
    root = w.session.root
    before = {p.name: p.read_text() for p in root.glob("*.json")}
    rows = _tick(d, lambda r: r.cells[0] == MISSING and r.cells[2] == "car")
    assert rows
    d.accept_rows(ask=False)
    changed = [p.name for p in root.glob("*.json") if p.read_text() != before[p.name]]
    assert len(changed) == len({r.key for r in rows}) and d.undo_btn.isEnabled()
    d.undo_accept()
    assert all(p.read_text() == before[p.name] for p in root.glob("*.json"))
    # reject a mismatch ("the label is right"): hidden now and on later runs; shown again with Show rejected
    rows = _tick(d, lambda r: r.cells[0] == MISMATCH)
    assert rows
    d.reject_rows()
    assert not any(r is not None and r.cells[0] == MISMATCH and r.key == rows[0].key for r in d.rows)
    d.show_rejected.setChecked(True)
    qtbot.waitUntil(lambda: any(r is not None and r.tone == "muted" for r in d.rows), timeout=20000)
    _tick(d, lambda r: r.tone == "muted")
    d.unreject_rows()
    qtbot.waitUntil(lambda: not any(r is not None and r.tone == "muted" for r in d.rows), timeout=20000)
    d.show_rejected.setChecked(False)
    # flag what the models disagree with
    rows = _tick(d, lambda r: r.cells[0] == MISSED)
    d.flag_files()
    from AIO_QC.core.state import REVIEW_FLAG
    assert all(w.session.review.status(r.key) == REVIEW_FLAG for r in rows)
    # bulk accept by confidence (its own small window — answered Cancel here)
    d.accept_by_confidence()
    # an audit: a sample of findings judged by hand → an estimate of the real error rate
    for cb in d.audit_kinds.values():
        cb.setChecked(True)
    d._create_audit()
    assert d.audit is not None and d.audit.items
    for v in ("error", "ok", None, "error"):
        d.audit_tbl.selectRow(0)
        d._set_verdict(v)
    d._open_audit_item(0)
    d._refresh_estimate()
    # the window reopens on the last run
    d.close()
    w.open_tool("model_check")
    d2 = w.open_dialogs["model_check"]
    qtbot.waitUntil(lambda: d2.explorer.state is not None or d2.ctx.get("eval") is not None, timeout=30000)
    d2.close()


def test_improve_and_sam_on_the_real_server(run, qtbot):
    from AIO_QC.core import improve as IM
    w, d = run
    ev = d.ctx["eval"]
    spots = IM.weak_spots(ev, 0.25, 0.5)
    assert {c["label"] for c in spots["classes"]} >= {"car", "plate"}
    hard = IM.hardest_images(ev, 0.25, 0.5, 5)
    assert hard and hard[0][1] >= hard[-1][1]
    img = str(next(w.session.root.glob("*.jpg")))
    polys = w.inference.segment("toy_sam.pt", img, [[40, 40, 140, 120], [200, 150, 260, 200]])
    assert len(polys) == 2 and len(polys[0]) >= 6 and len(polys[1]) >= 6
    xs = [p[0] for p in polys[0]]
    assert 40 <= min(xs) and max(xs) <= 140
