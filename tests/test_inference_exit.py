# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Closing an inference client while its reader threads still run (app exit) must not raise in those threads."""
import subprocess
import sys
import threading

import pytest


@pytest.mark.parametrize("modname", ["AIO_QC.core.inference", "AIO_QC.tools.frame_extractor.core.inference"])
def test_reader_survives_close(modname, monkeypatch):
    import importlib
    mod = importlib.import_module(modname)
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda a: errors.append(a.exc_value))
    c = mod.InferenceClient(sys.executable)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    c.proc = proc
    t = threading.Thread(target=c._read_stdout, args=(proc,), daemon=True)
    t.start()
    c.proc = None                                   # what close() does at the end
    t.join(5)
    proc.wait(5)
    assert not errors and c._dead and "exited" in c._dead
