# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Integration invariants that need no window: the import graph, and what AIO QC loads at startup."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "AIO_QC"
VENDORED = [PKG / "tools" / "frame_extractor", PKG / "core" / "testing", PKG / "widgets" / "testing"]
HEAVY = ("AIO_QC.tools.frame_extractor", "AIO_QC.core.testing", "AIO_QC.widgets.testing", "onnxruntime", "av",
         "imageio_ffmpeg", "pyarrow", "pandas")


def _py_files(roots):
    for r in roots:
        for p in r.rglob("*.py"):
            if "__pycache__" not in p.parts and "editors" not in p.relative_to(PKG).parts[:1]:
                yield p


def test_no_framextractor_imports_anywhere():
    """Mechanical vendoring proof: nothing imports the old top-level package; vendored code has no relative imports
    (each was resolved against its original location and rewritten absolute)."""
    bad = []
    for p in _py_files([PKG]):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        vendored = any(v in p.parents for v in VENDORED)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "FrameExtractor":
                    bad.append(f"{p}:{node.lineno} from {node.module}")
                if vendored and node.level:
                    bad.append(f"{p}:{node.lineno} relative import in vendored code")
            elif isinstance(node, ast.Import):
                bad += [f"{p}:{node.lineno} import {a.name}" for a in node.names
                        if a.name.split(".")[0] == "FrameExtractor"]
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and str(a.value).startswith("FrameExtractor"):
                    bad.append(f"{p}:{node.lineno} import_module({a.value!r})")
    assert not bad, "\n".join(bad)


def test_import_graph_layers():
    """Core stays Qt-free; the Testing core never imports widgets or the Frame Extractor GUI."""
    bad = []
    for p in _py_files([PKG / "core"]):
        rel = p.relative_to(PKG)
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.ImportFrom) and not node.level:
                mods = [node.module or ""]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if rel.parts[:2] == ("core", "testing") and (
                        m.startswith(("AIO_QC.widgets", "AIO_QC.tools.frame_extractor.widgets",
                                      "AIO_QC.tools.frame_extractor.app"))):
                    bad.append(f"{rel}:{node.lineno} imports GUI module {m}")
    assert not bad, "\n".join(bad)


def _run(code: str) -> dict:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(PKG.parent))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180,
                         cwd=str(PKG.parent))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_frame_extractor_not_loaded_at_startup(tmp_path):
    """Starting AIO QC (the shell + the Image QC window) loads neither the Frame Extractor, Testing, nor the video /
    model / data libraries — they load on first use."""
    code = f"""
import os, sys, json
os.environ["HOME"] = {str(tmp_path)!r}
os.environ["XDG_CONFIG_HOME"] = {str(tmp_path / "cfg")!r}
from PyQt6.QtWidgets import QApplication
app = QApplication(["t"])
from AIO_QC.shell import AppShell
sh = AppShell(app)
app.main_window = sh
w = sh.show("image")
for _ in range(20):
    app.processEvents()
print(json.dumps(sorted(m for m in sys.modules if m.startswith({HEAVY!r}))))
"""
    loaded = _run(code)
    assert loaded == [], f"loaded at startup: {loaded}"


def test_cli_subcommands_are_qt_free():
    code = """
import sys, json, contextlib, io
from AIO_QC.tools.frame_extractor.core import cli
with contextlib.redirect_stdout(io.StringIO()):
    try:
        cli.main(["probe", "--help"])
    except SystemExit:
        pass
print(json.dumps([m for m in sys.modules if m.startswith("PyQt6")]))
"""
    assert _run(code) == []


def test_python_310_syntax():
    """pyproject says Python >= 3.10: no 3.12-only syntax (e.g. reused quotes inside f-strings) anywhere."""
    import shutil
    import pytest
    ruff = shutil.which("ruff")
    if not ruff:
        pytest.skip("ruff not installed")
    r = subprocess.run([ruff, "check", "--no-cache", "--target-version", "py310", "--select", "E9",
                        "--exclude", "AIO_QC/editors", str(PKG)], capture_output=True, text=True, cwd=str(PKG.parent))
    assert r.returncode == 0, r.stdout[-3000:]


def test_old_app_gpu_and_crash_state_is_not_migrated(tmp_path):
    """The standalone app's 'GPU failed' / crash sentinel must never turn the GPU off in AIO QC, and a 2.1.0
    install that remembered 'GPU failed' gets one fresh GPU attempt."""
    from PyQt6.QtCore import QSettings
    from AIO_QC.utils import fx_settings as F
    for k in ("ui/gpu_failed", "crash/sentinel", "safe/last_crash", "ui/theme"):
        assert F.map_key(k) is None
    assert F.map_key("ui/renderer") == "fx/ui/renderer" and F.map_key("lab/x") == "testing/x"
    s = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    s.setValue("fx/ui/gpu_failed", "layered window")
    s.setValue("fx/crash/sentinel", "gpu")
    s.setValue("fx/ui/renderer", "gpu")
    F._gpu_retry_once(s)
    assert not s.contains("fx/ui/gpu_failed") and not s.contains("fx/crash/sentinel")
    assert s.value("fx/ui/renderer") == "gpu"
    s.setValue("fx/ui/gpu_failed", "real failure")
    F._gpu_retry_once(s)                                 # only once: a real later failure is remembered
    assert s.value("fx/ui/gpu_failed") == "real failure"
