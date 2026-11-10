# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Process-wide pieces exist once; settings and files of the standalone FrameExtractor move over once."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_settings_migration(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from AIO_QC.utils import fx_settings as FS
    import AIO_QC.utils.qt_utils as QU
    new_ini = tmp_path / "aio.ini"
    monkeypatch.setattr(QU, "app_settings", lambda: QSettings(str(new_ini), QSettings.Format.IniFormat))
    monkeypatch.setattr(FS, "app_settings", QU.app_settings)
    monkeypatch.setattr(FS, "_checked", [False])
    old = QSettings(str(tmp_path / "fx.ini"), QSettings.Format.IniFormat)
    for k, v in {"output_dir": "/data/out", "lab/testset": "/t/ts", "lab/batch": 16, "ui/theme": "dark",
                 "ui/animations": "reduced", "models/paths": ["/m/a.onnx"], "playback/libmpv": "/x/libmpv.so"}.items():
        old.setValue(k, v)
    old.sync()
    pre = QU.app_settings()
    pre.setValue("fx/output_dir", "/already/here")              # an existing AIO QC value always wins
    pre.sync()
    n = FS.migrate_settings(old)
    s = QU.app_settings()
    assert n == 5
    assert s.value("fx/output_dir") == "/already/here"
    assert s.value("testing/testset") == "/t/ts" and int(s.value("testing/batch")) == 16
    assert s.value("ui/animations") == "reduced"
    assert not s.contains("fx/ui/theme") and not s.contains("ui/theme")
    assert s.value("fx/playback/libmpv") == "/x/libmpv.so"
    monkeypatch.setattr(FS, "_checked", [False])
    assert FS.migrate_settings(old) == 0                          # once
    t = FS.testing_settings()
    assert t.value("testset") == "/t/ts"                          # the Testing view of the store
    f = FS.fx_settings()
    f.setValue("probe", 1)
    assert QU.app_settings().value("fx/probe") is not None


def test_file_migration(tmp_path, monkeypatch):
    from AIO_QC.core import fx_storage as S
    old = tmp_path / "home" / ".frame_extractor"
    for rel, txt in {"sessions/a.json": "{}", "search/idx.npy": "x", "lab/recent.json": "[]",
                     "lab/exclusions/e.json": "[]", "lab/preds.sqlite": "db", "logs/frame_extractor.log": "log"}.items():
        p = old / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    data = tmp_path / "data"
    monkeypatch.setattr(S, "data_home", lambda: data)
    monkeypatch.setattr(S, "_done", [False])
    (data / "testing").mkdir(parents=True)
    (data / "testing" / "recent.json").write_text('["mine"]')     # never overwritten
    r = S.migrate_files(old)
    assert (data / "frame_extractor" / "sessions" / "a.json").is_file()
    assert (data / "frame_extractor" / "search" / "idx.npy").is_file()
    assert (data / "testing" / "exclusions" / "e.json").is_file()
    assert (data / "testing" / "preds.sqlite").read_text() == "db"
    assert (data / "testing" / "recent.json").read_text() == '["mine"]'
    assert (old / "logs" / "frame_extractor.log").is_file()           # logs stay
    assert (old / "MOVED_TO_AIO_QC.txt").is_file() and (data / S.MARKER).is_file()
    assert any("recent.json" in k for k in r["kept"])
    monkeypatch.setattr(S, "_done", [False])
    assert S.migrate_files(old)["skipped"]                             # once


def test_process_setup_exactly_once(tmp_path):
    code = """
import json, locale, sys
from AIO_QC import __main__ as M
M._process_setup_before_qt()
from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtGui import QSurfaceFormat
app = QApplication(["t"])
M._process_setup_after_qt(app)
M._process_setup_after_qt(app)                     # a second call must not add a second filter
from AIO_QC.widgets import motion
from AIO_QC.widgets.motion import AppMotion
n = sum(isinstance(c, AppMotion) for c in app.children())
d = QDialog(); d.resize(20000, 20000); d.show(); app.processEvents()
g = app.primaryScreen().availableGeometry()
print(json.dumps({"motion": n, "numeric": locale.setlocale(locale.LC_NUMERIC), "swap": QSurfaceFormat.defaultFormat().swapInterval(),
                  "fits": d.width() <= g.width() and d.height() <= g.height()}))
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "c"),
               PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["motion"] == 1 and r["swap"] == 1 and r["fits"]
    if sys.platform != "win32":
        assert r["numeric"] == "C"


def test_loguru_sinks_not_added_by_frame_extractor():
    from loguru import logger
    from AIO_QC.tools.frame_extractor.core import crash
    before = len(logger._core.handlers)
    crash.setup_logging("DEBUG")
    crash.install_exception_hooks()
    crash.enable_native_crash_log()
    assert len(logger._core.handlers) == before
