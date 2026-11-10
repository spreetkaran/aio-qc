# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The model servers run inside the user's Python environment as `python <server>.py`. Python then puts the server's
own folder first on sys.path, and the app's modules there (core/profile.py, …) shadowed the standard library:
torch imports cProfile, cProfile imports `profile`, got AIO QC's, and ultralytics' lazy import turned that into
"cannot import name 'YOLO' from 'ultralytics'". These tests start the real servers with a stand-in ultralytics
that does exactly what torch does."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVERS = [ROOT / "AIO_QC" / "core" / "infer_server.py", ROOT / "AIO_QC" / "core" / "embed_server.py",
           ROOT / "AIO_QC" / "core" / "ocr_server.py",
           ROOT / "AIO_QC" / "tools" / "frame_extractor" / "core" / "infer_server.py"]

FAKE_ULTRALYTICS = '''
__all__ = ("YOLO", "YOLOE")


def __getattr__(name):                       # lazy, like ultralytics 8.3: errors inside become "cannot import name"
    if name == "YOLO":
        import cProfile                      # what torch (torch._dynamo) does on import
        cProfile.run                         # fails when `profile` is not the standard library's
        from ultralytics._yolo import YOLO
        return YOLO
    raise AttributeError(name)
'''
FAKE_YOLO = '''
class YOLO:
    def __init__(self, path):
        self.names = {0: "car", 1: "plate"}
        self.task = "detect"
        self.overrides = {"imgsz": 640}

    def predict(self, *a, **k):
        return []
'''


@pytest.mark.parametrize("server,prefix", [(SERVERS[0], "@@AIOQC@@ "), (SERVERS[3], "@@FXINFER@@ ")])
def test_infer_server_loads_ultralytics_despite_app_modules_next_to_it(tmp_path, server, prefix):
    pkg = tmp_path / "site" / "ultralytics"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_ULTRALYTICS)
    (pkg / "_yolo.py").write_text(FAKE_YOLO)
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "site"))
    req = [{"rid": 1, "cmd": "load", "models": [{"id": "m", "path": str(weights), "imgsz": 640, "device": "cpu",
                                                  "half": False, "sahi": None, "warmup": False}]},
           {"rid": 2, "cmd": "shutdown"}]
    r = subprocess.run([sys.executable, "-u", str(server)], input="\n".join(json.dumps(x) for x in req) + "\n",
                       capture_output=True, text=True, env=env, timeout=60, cwd=str(server.parent))
    replies = [json.loads(line[len(prefix):]) for line in r.stdout.splitlines() if line.startswith(prefix)]
    first = next(x for x in replies if x.get("rid") == 1)
    models = first.get("models") or first.get("result") or first
    blob = json.dumps(models)
    assert '"ok": true' in blob and "car" in blob, (blob, r.stderr[-2000:])
    assert "cannot import name" not in r.stderr


@pytest.mark.parametrize("server", SERVERS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_server_leaves_its_folder_off_the_path_first(server):
    src = server.read_text(encoding="utf-8").split("\n")
    first_import = next(i for i, l in enumerate(src) if l.startswith("import ") or l.startswith("from "))
    block = "\n".join(src[first_import:first_import + 12])
    assert "_sys.path[:] = [p for p in _sys.path" in block, server
    # and the shadowing modules are still there, so the guard matters
    names = {p.stem for p in server.parent.glob("*.py")}
    assert names & {"profile", "io", "imageio"}
