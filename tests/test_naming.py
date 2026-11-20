# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The renaming engine (core.naming) and its panel: presets, your own pattern, verification (case-only clashes, one
label file for a.jpg and a.png, files already there, what Windows refuses, path length), numbering the rest,
renamed.csv — and a million names in seconds."""
import time

import pytest

from AIO_QC.core import naming as N


def _items(*rels):
    return [N.Item(r, r) for r in rels]


def test_clashes_ignore_case_and_compare_stems():
    its = _items("camA/img001.jpg", "camB/IMG001.png", "camA/a.jpg", "camA/a.png", "camC/unique.jpg")
    flat = sorted(sorted(g) for g in N.clashes(its, flat=True))
    assert flat == [["camA/a.jpg", "camA/a.png"], ["camA/img001.jpg", "camB/IMG001.png"]]
    kept = N.clashes(its, flat=False)                           # sub-folders kept: only a.jpg / a.png share a.json
    assert [sorted(g) for g in kept] == [["camA/a.jpg", "camA/a.png"]]


@pytest.mark.parametrize("preset,expect", [
    ("folder", {"camA/x/img.jpg": "x__img", "camB/x/img.jpg": "x__img"}),
    ("path", {"camA/x/img.jpg": "camA_x__img", "camB/x/img.jpg": "camB_x__img"}),
    ("sequence", {"camA/x/img.jpg": "000001", "camB/x/img.jpg": "000002"}),
    ("suffix", {"camA/x/img.jpg": "img", "camB/x/img.jpg": "img_1"}),
])
def test_presets(preset, expect):
    P = N.plan(_items("camA/x/img.jpg", "camB/x/img.jpg", "camA/other.jpg"), N.Scheme(preset=preset))
    assert {k: P.new[k] for k in expect} == expect
    assert P.new["camA/other.jpg"] == "other"                   # does not clash: left alone (the default)
    assert P.ok == (preset != "folder")                         # x__img twice still clashes: shown, not hidden


def test_folder_preset_still_clashing_is_numbered():
    P = N.plan(_items("camA/x/img.jpg", "camB/x/img.jpg"), N.Scheme(preset="folder"))
    assert not P.ok and P.clash_count() == 2 and "clashes with x__img" in P.problems["camA/x/img.jpg"][0]
    P = N.resolve(P)
    assert P.ok and sorted(P.new.values()) == ["x__img", "x__img_1"] and "ready" in P.summary()


def test_every_file_and_custom_pattern(tmp_path):
    f = tmp_path / "a" / "pic one.jpg"
    f.parent.mkdir()
    f.write_bytes(b"123")
    its = [N.Item("k", "a/pic one.jpg", split="train", path=str(f))]
    sc = N.Scheme(preset="custom", pattern="{split}-{folder}-{stem}-{n:03d}-{hash8}", only_clashes=False,
                  case="lower", start=7)
    P = N.plan(its, sc)
    assert P.new["k"] == f"train-a-pic_one-007-{N.file_hash8(str(f))}" and P.renamed == 1
    assert N.bad_tokens("{stem}_{nope}") == ["nope"] and N.bad_tokens("{stem") and N.bad_tokens("{stem}{n:04d}") == []
    assert N.Scheme.from_dict(sc.to_dict()) == sc


def test_windows_rules():
    its = _items("con.jpg", "ok.jpg", "bad?name.jpg", "dot..jpg")
    P = N.plan(its, N.Scheme(preset="custom", pattern="{stem}", only_clashes=False, spaces=False))
    assert any("reserves (CON)" in p for p in P.problems["con.jpg"])
    assert "ok.jpg" not in P.problems and "bad?name.jpg" not in P.problems    # (the ? is cleaned while renaming)
    assert any("ends in a dot" in p for p in P.problems["dot..jpg"])
    P = N.resolve(P)
    assert P.ok and P.new["con.jpg"] == "con_file" and P.new["dot..jpg"] == "dot"


def test_path_length_and_existing_files():
    its = [N.Item("k", "x/" + "n" * 120 + ".jpg", extra_exts=(".json", ".txt"))]
    P = N.plan(its, N.Scheme(preset="folder", only_clashes=False), root_len=150)
    assert any("too long for Windows" in p for p in P.problems["k"])
    assert N.resolve(P, root_len=150).ok
    P = N.plan(_items("a/img.jpg"), N.Scheme(preset="folder", only_clashes=False), existing={"": {"a__img"}})
    assert "a file already there" in P.problems["a/img.jpg"][0]
    P = N.resolve(P, existing={"": {"a__img"}})
    assert P.ok and P.new["a/img.jpg"] == "a__img_1"


def test_renamed_csv_round_trip(tmp_path):
    P = N.resolve(N.plan(_items("camA/i.jpg", "camB/i.jpg", "solo.jpg"), N.Scheme(preset="folder")))
    N.write_csv(tmp_path / "renamed.csv", P, root="/data")
    back = N.read_csv(tmp_path / "renamed.csv")
    assert back == {"camA__i.jpg": "/data/camA/i.jpg", "camB__i.jpg": "/data/camB/i.jpg"}
    assert P.new_rel("solo.jpg") == "solo.jpg"


def test_a_million_names_in_seconds():
    its = [N.Item(f"k{i}", f"cam{i % 50}/day{i % 7}/img{i // 350:06d}.jpg") for i in range(1_000_000)]
    t = time.monotonic()
    P = N.resolve(N.plan(its, N.Scheme(preset="folder")))
    took = time.monotonic() - t
    assert P.ok and took < 90, took


def test_panel_live_verification_presets_and_fix(qtbot):
    from AIO_QC.widgets import naming_panel as NP
    NP.save_user_presets([])
    p = NP.NamingPanel(context="test")
    qtbot.addWidget(p)
    p.set_items(_items("camA/x/img.jpg", "camB/x/img.jpg", "other.jpg"), flat=True)
    i = next(n for n in range(p.preset.count()) if (p.preset.itemData(n) or {}).get("builtin") == "folder")
    p.preset.setCurrentIndex(i)
    P = p.wait()
    assert P is not None and not P.ok and not p.fix_btn.isHidden() and "clashes" in p.summary.text()
    assert p.table.rowCount() >= 2 and p.table.item(0, 2).text()
    p.fix_btn.click()
    assert p.is_ready() and p.fix_btn.isHidden() and "ready" in p.summary.text()
    # your own pattern: a token that does not exist is shown at once, nothing goes ahead
    p.pattern.setText("{folder}-{oops}")
    p._pattern_edited(p.pattern.text())
    assert "oops" in p.summary.text() and p.plan() is None
    p.pattern.setText("{path}{sep}{stem}")
    p._pattern_edited(p.pattern.text())
    assert p.wait().ok and p.scheme().preset == "path"            # typed, but it is that preset
    # saved as a preset of yours, chosen again later
    from PyQt6.QtWidgets import QInputDialog
    orig = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("mine", True))
    try:
        p._save_preset()
    finally:
        QInputDialog.getText = orig
    assert [x["name"] for x in NP.user_presets()] == ["mine"] and "mine" in p.preset.currentText()
    p.preset.setCurrentIndex(0)
    p.preset.setCurrentIndex(p.preset.count() - 1)
    assert p.pattern.text() == "{path}{sep}{stem}"
    p._delete_preset()
    assert NP.user_presets() == []
    p.wait()
    # remembered per place it is used
    q = NP.NamingPanel(context="test")
    qtbot.addWidget(q)
    assert q.pattern.text() == p.pattern.text()
