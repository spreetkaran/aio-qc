# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""`aio-qc extract / probe / testset` from the command line, on a small video made here."""
import json

import pytest

from AIO_QC.tools.frame_extractor.core import cli

from .test_fx_extract_core import _video


def test_probe_and_errors(tmp_path, capsys):
    v = _video(tmp_path / "v.avi", n=20)
    assert cli.main(["probe", str(v), str(tmp_path / "missing.avi")]) == 0
    out = capsys.readouterr().out
    assert "160x120" in out and "20 frames" in out and "missing.avi: ERROR" in out
    assert cli.main(["extract", str(tmp_path / "nothing_here"), "-o", str(tmp_path / "o")]) == 2
    assert cli.main(["extract", str(v), "-o", str(tmp_path / "o"), "--track-roi"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["extract", str(v), "-o", str(tmp_path / "o"), "--roi", "1,2,3"])
    assert cli._parse_frames("10,20") == [[10, None], [20, None]]       # an ROI per frame: only in an @file


def test_extract_with_preset_and_markers(tmp_path, capsys):
    vids = tmp_path / "vids"
    _video(vids / "a.avi", n=30)
    _video(vids / "sub" / "b.avi", n=30)
    out = tmp_path / "out"
    preset = tmp_path / "p.json"
    code = cli.main(["extract", str(vids), "-r", "-o", str(out), "--every", "1s", "--mode", "sharpest",
                     "--dedup", "--format", "png", "--save-preset", str(preset), "-w", "1", "--no-split"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "2 video(s)" in text and "Preset saved" in text
    p = json.loads(preset.read_text())
    assert p["mode"] == "sharpest" and p["dedup"] and p["ext"] == ".png"
    assert len(list((out / "a" / "batch_extract").glob("*.png"))) == 3
    assert len(list((out / "b" / "batch_extract").glob("*.png"))) == 3
    # from the preset, exact frames → the markers folder
    lst = tmp_path / "frames.txt"
    lst.write_text("3\n7 0,0,40,40\n")
    code = cli.main(["extract", str(vids / "a.avi"), "-o", str(out), "--preset", str(preset), "--frames", f"@{lst}",
                     "-w", "1"])
    assert code == 0
    assert len(list((out / "a" / "markers").glob("*.png"))) == 2


def test_testset_from_a_video(tmp_path, capsys):
    v = _video(tmp_path / "v.avi", n=40)
    ts = tmp_path / "ts"
    assert cli.main(["testset", str(ts), "--add", str(v), str(tmp_path / "nope"), "--every", "10f",
                     "--unique", "off"]) == 0
    out = capsys.readouterr()
    assert "+4 images" in out.out and "not found" in out.err
    assert cli.main(["testset", str(ts), "--add", str(v), "--every", "10f", "--unique", "exact"]) == 0
    assert "+0 images" in capsys.readouterr().out                     # the same frames: already in the set
