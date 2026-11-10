# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Every source file carries the copyright / licence header, and LICENSE is the GPL-3.0 text."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_spdx_headers_and_license():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "packaging", "add_spdx.py"), ROOT, "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    text = open(os.path.join(ROOT, "LICENSE"), encoding="utf-8").read()
    assert text.lstrip().startswith("GNU GENERAL PUBLIC LICENSE") and "Version 3, 29 June 2007" in text
