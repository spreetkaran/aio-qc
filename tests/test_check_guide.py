# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Recommended checks (core ranking + Image QC wiring) and where saves go (utils.save_dirs)."""
import json
from collections import Counter
from pathlib import Path

import pytest

from AIO_QC.core import check_guide as CG
from tests import fixtures as FX
from tests.test_aio_image import _open, shell  # noqa: F401  (fixture)


# ------------------------------------------------------------------ ranking (Qt-free)
def test_unreadable_files_come_first():
    recs = CG.recommend(CG.Signals(files=100, unreadable=3, labels=Counter(car=50, bus=40)))
    assert recs[0].tool == "validation" and "3 annotation" in recs[0].why


def test_label_variants_and_unknown_labels():
    assert CG.label_variants(["Car ", "car", "c_ar", "bus"]) == [["Car ", "c_ar", "car"]]
    recs = CG.recommend(CG.Signals(files=50, labels=Counter({"car": 30, "Car": 3, "bus": 9})))
    top = [r.tool for r in recs[:2]]
    assert "ambiguity" in top
    recs = CG.recommend(CG.Signals(files=50, labels=Counter(car=30, truck=2), class_list=["car", "bus"]))
    amb = next(r for r in recs if r.tool == "ambiguity")
    assert amb.score >= 90 and "truck" in amb.why


def test_frames_relations_models_and_order():
    s = CG.Signals(files=500, labels=Counter(car=400, plate=380), frame_like=0.9,
                   parts=[("plate", "car", 0.99)], models=1, texts=40, empty=20)
    recs = CG.recommend(s)
    tools = [r.tool for r in recs]
    assert tools.index("image_dups") < tools.index("duplicates")
    assert "sequence" in tools and "rules" in tools and "text_labels" in tools and "model_check" in tools
    rules = next(r for r in recs if r.tool == "rules")
    assert "plate" in rules.why and "car" in rules.why
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_done_checks_move_last_and_next_advances():
    s = CG.Signals(files=10, unreadable=1, labels=Counter(a=5, b=5))
    recs = CG.recommend(s, {"validation": 4})
    assert recs[-1].tool == "validation" and recs[-1].done and recs[-1].found == 4
    assert CG.next_check(recs).tool != "validation"
    every = {r.tool: 0 for r in recs}
    assert CG.next_check(CG.recommend(s, every)) is None


def test_frame_share():
    assert CG.frame_share(["cam_frame_0001.jpg", "img_0002.png", "street.jpg", "x-12345.jpg"]) == 0.75
    assert CG.frame_share([]) == 0.0


# ------------------------------------------------------------------ Image QC
def test_guide_on_open_and_after_a_check(shell, qtbot, tmp_path):  # noqa: F811
    root = tmp_path / "ds"
    FX.labelme(root, 30)
    w = _open(shell, qtbot, root)
    g = w.check_guide
    qtbot.waitUntil(lambda: g._sample is not None, timeout=20000)
    assert g.recs and g.next() is not None
    assert g._sample["frame_like"] == 1.0                     # cam_frame_0000 …
    assert not g.btn.isHidden() and "Next check" in g.btn.full
    assert w.toast.find("check_guide") is not None
    first = g.next().tool
    g.record(first, 5)
    assert g.next().tool != first
    saved = json.loads((root / ".aio_qc" / "guide.json").read_text())
    assert saved["done"][first] == 5
    # a real scan records itself
    w.open_tool("empty")
    dlg = w.open_dialogs["empty"]
    dlg.start_btn.click()
    qtbot.waitUntil(lambda: dlg.worker is None and "empty" in g.done, timeout=20000)
    # setting off: no button, no hint
    w.settings.setValue("guide/enabled", False)
    g.setting_changed(False)
    assert g.btn.isHidden()
    w.settings.setValue("guide/enabled", True)
    g.setting_changed(True)
    assert not g.btn.isHidden()
    g.forget()
    assert g.done == {}


def test_guide_setting_listed(shell, qtbot, tmp_path):  # noqa: F811
    from AIO_QC.widgets.settings_pages import image_sections
    w = shell.window("image")
    keys = {r.key for sec in image_sections(w) for r in sec.settings}
    assert "image.check_guide" in keys and "app.save_screenshots" in keys and "app.save_default" in keys


# ------------------------------------------------------------------ save locations
class _St(dict):
    def value(self, k, d=None, type=None):
        v = self.get(k, d)
        return type(v) if type is not None and v is not None else v

    def setValue(self, k, v):
        self[k] = v

    def remove(self, k):
        self.pop(k, None)


def test_save_dirs_priority(tmp_path, qapp):
    from AIO_QC.utils import save_dirs as SD
    st = _St()
    fb = tmp_path / "next_to_dataset"
    fb.mkdir()
    assert SD.start_dir("exports", fb, st) == fb                         # nothing set: the feature's folder
    base = tmp_path / "saves"
    base.mkdir()
    st[SD.DEFAULT_KEY] = str(base)
    assert SD.start_dir("exports", fb, st) == base                       # default folder for saves
    own = tmp_path / "reports"
    own.mkdir()
    st[SD.KEY["reports"]] = str(own)
    assert SD.start_dir("reports", fb, st) == own                        # the kind's own folder
    last = tmp_path / "last"
    last.mkdir()
    SD.remember("reports", last / "r.html", st)
    assert SD.start_dir("reports", fb, st) == last                       # where I saved last
    st[SD.REMEMBER_KEY] = False
    assert SD.start_dir("reports", fb, st) == own
    assert SD.folder_for("screenshots", st) == base / "AIO QC Screenshots"
    st[SD.KEY["screenshots"]] = str(tmp_path / "shots")
    assert SD.folder_for("screenshots", st) == Path(tmp_path / "shots")


def test_ask_save_starts_in_folder_and_remembers(tmp_path, qapp, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from AIO_QC.utils import save_dirs as SD
    st = _St()
    monkeypatch.setattr(SD, "_settings", lambda: st)
    out = tmp_path / "out"
    out.mkdir()
    st[SD.KEY["exports"]] = str(out)
    seen = {}

    def fake(parent, cap, start, filt):
        seen["start"] = start
        return str(tmp_path / "picked" / "x.csv"), filt
    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake)
    (tmp_path / "picked").mkdir()
    path, _ = SD.ask_save(None, "Export", "issues.csv", "CSV (*.csv)", kind="exports")
    assert seen["start"] == str(out / "issues.csv")
    assert st["saves/last/exports"] == str(tmp_path / "picked")
    SD.ask_save(None, "Export", "again.csv", "CSV (*.csv)", kind="exports")
    assert seen["start"] == str(tmp_path / "picked" / "again.csv")


@pytest.mark.parametrize("kind", list(__import__("AIO_QC.utils.save_dirs", fromlist=["KINDS"]).KINDS))
def test_every_kind_has_a_setting(kind, qapp):
    from AIO_QC.utils import save_dirs as SD
    keys = {r.key for r in SD.settings_section().settings}
    assert f"app.save_{kind}" in keys
