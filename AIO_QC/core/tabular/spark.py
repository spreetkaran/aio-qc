# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Running a recipe on Spark (Qt-free): the generated PySpark script is run by the Python environment you choose (one
that has pyspark), locally (local[*]), on a cluster (--master yarn / spark://…) or through Spark Connect
(--remote sc://…). The app never imports pyspark itself.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

PROBE = ("import json, os, shutil, sys\n"
         "out = {'python': sys.version.split()[0]}\n"
         "try:\n    import pyspark; out['pyspark'] = pyspark.__version__\nexcept Exception as e:\n"
         "    out['pyspark'] = None; out['error'] = str(e)\n"
         "out['java'] = bool(os.environ.get('JAVA_HOME') or shutil.which('java'))\n"
         "print(json.dumps(out))\n")


def probe(python: str, timeout: float = 60) -> dict:
    """{'python', 'pyspark' (version or None), 'java' (bool), 'error'?}"""
    import json
    flags = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
    r = subprocess.run([python, "-c", PROBE], capture_output=True, text=True, timeout=timeout, **flags)
    if r.returncode != 0:
        return {"pyspark": None, "java": False, "error": (r.stderr or r.stdout).strip()[-400:]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"pyspark": None, "java": False, "error": r.stdout[-400:]}


def write_script(code: str, folder: str | None = None) -> str:
    d = folder or tempfile.mkdtemp(prefix="aioqc_spark_")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "recipe_pyspark.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(code)
    return p


def command(python: str, script: str, master: str = "local[*]", remote: str = "") -> list[str]:
    cmd = [python, "-u", script]
    if remote.strip():
        cmd += ["--remote", remote.strip()]
    elif master.strip():
        cmd += ["--master", master.strip()]
    return cmd


def has_java() -> bool:
    return bool(os.environ.get("JAVA_HOME") or shutil.which("java"))
