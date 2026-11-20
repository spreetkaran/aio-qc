# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The toolbar path keeps the file name when files appear or disappear outside AIO QC (3.0.0 fix)."""
import shutil
from pathlib import Path

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def test_file_name_survives_outside_changes(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    qtbot.waitUntil(lambda: w.current_key is not None)
    bc = w.dataset_label
    qtbot.waitUntil(lambda: bc._file != "")
    shown = bc._file
    n0 = len(w.session.samples)
    src = sorted(Path(root).rglob("*.json"))[-1]
    img = sorted(Path(root).rglob("*.jpg"))[-1]
    shutil.copy(img, img.with_name("zz_new.jpg"))
    img.with_name("zz_new.json").write_text(src.read_text().replace(img.name, "zz_new.jpg"))
    qtbot.waitUntil(lambda: len(w.session.samples) == n0 + 1, timeout=15000)
    assert bc._file == shown
    assert f"{n0 + 1:,} files" in bc.pill.text()
    img.with_name("zz_new.json").unlink()
    img.with_name("zz_new.jpg").unlink()
    qtbot.waitUntil(lambda: len(w.session.samples) == n0, timeout=15000)
    assert bc._file == shown
    assert f"{n0:,} files" in bc.pill.text()
