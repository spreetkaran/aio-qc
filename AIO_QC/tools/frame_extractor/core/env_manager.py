# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Python environments for inference: discovery, probing, and temporary environments.
(From AIO QC; adds quick_check / find_ready_env for one-click setup.)

A temporary environment is a venv in <tmp>/aio_qc_envs/<id>; the install is a list of commands the GUI
runs one by one (streaming output into its console) and the folder is deleted after use.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from AIO_QC.core import procs



TEMP_ROOT = Path(tempfile.gettempdir()) / "frameextractor_envs"
TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
}
TORCH_CHOICES = [
    ("auto", "Auto (detect GPU driver)"),
    ("default", "PyPI default (Linux: CUDA · Windows/macOS: CPU/MPS)"),
    ("cpu", "CPU only (smallest download)"),
    ("cu118", "CUDA 11.8"), ("cu121", "CUDA 12.1"), ("cu124", "CUDA 12.4"), ("cu126", "CUDA 12.6"),
]


def python_in_env(path: str | Path) -> Path | None:
    """Accepts an env folder (conda/venv) or a python executable."""
    p = Path(path)
    if p.is_file():
        return p
    cands = ([p / "python.exe", p / "Scripts" / "python.exe"] if sys.platform == "win32"
             else [p / "bin" / "python3", p / "bin" / "python"])
    return next((c for c in cands if c.is_file()), None)


def _run(cmd, timeout=20) -> str:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        return procs.run(cmd, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def discover_envs() -> list[tuple[str, str]]:
    """[(display name, python executable)] — this app's Python, conda envs, common env folders."""
    found: dict[str, str] = {}

    def add(name, py):
        if py and Path(py).is_file():
            key = os.path.normcase(os.path.abspath(py))
            found.setdefault(key, name)

    add(f"This app's Python ({Path(sys.executable).parent.name})", sys.executable)
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")
    if conda:
        try:
            for env in json.loads(_run([conda, "env", "list", "--json"]) or "{}").get("envs", []):
                py = python_in_env(env)
                add(f"conda: {Path(env).name}", str(py) if py else None)
        except ValueError:
            pass
    home = Path.home()
    for base in ("miniconda3", "anaconda3", "mambaforge", "miniforge3", ".conda"):
        root = home / base
        py = python_in_env(root)
        if py:
            add(f"conda: {base} (base)", str(py))
        envs = root / "envs"
        if envs.is_dir():
            for e in sorted(envs.iterdir()):
                py = python_in_env(e)
                add(f"conda: {e.name}", str(py) if py else None)
    for base in (home / ".virtualenvs", home / "venvs", home / ".venvs"):
        if base.is_dir():
            for e in sorted(base.iterdir()):
                py = python_in_env(e)
                add(f"venv: {e.name}", str(py) if py else None)
    for name, py in (("system python3", shutil.which("python3")), ("system python", shutil.which("python"))):
        add(name, py)
    return [(name, key) for key, name in found.items()]


def detect_cuda_driver() -> str | None:
    """Highest CUDA runtime the NVIDIA driver supports (from nvidia-smi), e.g. '12.4'."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", _run([exe], timeout=10))
    return m.group(1) if m else None


def pick_torch_build(driver: str | None) -> str:
    if not driver:
        return "cpu" if sys.platform != "darwin" else "default"
    v = tuple(int(x) for x in driver.split(".")[:2])
    for build, need in (("cu126", (12, 6)), ("cu124", (12, 4)), ("cu121", (12, 1)), ("cu118", (11, 8))):
        if v >= need:
            return build
    return "cpu"


def new_temp_env_dir() -> Path:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TEMP_ROOT / f"env_{uuid.uuid4().hex[:8]}"


def env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def install_plan(env_dir: Path, packages: list[str], torch_build: str | None, keep_pip_cache: bool = True,
                 base_python: str | None = None, inherit: bool = False) -> list[tuple[str, list[str]]]:
    """
    [(step title, argv)] — executed in order by the GUI.
    inherit=True creates the venv with --system-site-packages on top of `base_python`, so an existing
    PyTorch/CUDA install is reused and only the missing packages (e.g. ultralytics, sahi) are downloaded.
    torch_build=None skips the PyTorch step.
    """
    py = str(env_python(env_dir))
    pip = [py, "-m", "pip", "install", "--disable-pip-version-check"]
    if not keep_pip_cache:
        pip.append("--no-cache-dir")
    venv = [base_python or sys.executable, "-m", "venv"] + (["--system-site-packages"] if inherit else []) + [str(env_dir)]
    steps = [("Create virtual environment" + (" (reusing the base environment's packages)" if inherit else ""), venv),
             ("Upgrade pip", pip + ["--upgrade", "pip"])]
    if torch_build:
        cmd = pip + ["torch", "torchvision"]
        if torch_build in TORCH_INDEX:
            cmd += ["--index-url", TORCH_INDEX[torch_build]]
        steps.append((f"Install PyTorch ({torch_build})", cmd))
    if packages:
        steps.append(("Install " + " + ".join(packages), pip + list(packages)))
    return steps


def required_packages(mode: str) -> list[str]:
    return ["sahi", "ultralytics"] if mode == "sahi" else ["ultralytics"]


def missing_packages(info: dict | None, mode: str) -> list[str]:
    """What the probed environment lacks for the chosen mode (torch listed separately)."""
    if not info:
        return ["torch"] + required_packages(mode)
    miss = [] if info.get("torch") else ["torch"]
    miss += [p for p in required_packages(mode) if not info.get(p)]
    return miss


def _on_rm_error(func, path, _exc):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def remove_env(env_dir: Path) -> bool:
    """Delete a temporary env (only inside TEMP_ROOT, as a safety net)."""
    env_dir = Path(env_dir)
    try:
        env_dir.resolve().relative_to(TEMP_ROOT.resolve())
    except ValueError:
        return False
    shutil.rmtree(env_dir, onerror=_on_rm_error)
    return not env_dir.exists()


def dir_size_mb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2**20


_QUICK = ("import importlib.util as u, json, sys; "
          "d = {m: u.find_spec(m) is not None for m in ('ultralytics', 'torch', 'sahi')}; "
          "d['python'] = sys.version.split()[0]; print(json.dumps(d))")


def quick_check(python: str, timeout=20) -> dict | None:
    """Which packages an environment has, WITHOUT importing them (seconds, not minutes)."""
    out = _run([python, "-c", _QUICK], timeout=timeout)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def find_ready_env(progress=None, cancel=None):
    """First discovered environment that has ultralytics + torch. Returns (name, python) or None.
    progress(i, n, name) is called per candidate."""
    envs = discover_envs()
    for i, (name, py) in enumerate(envs):
        if cancel is not None and cancel():
            return None
        if progress:
            progress(i, len(envs), name)
        info = quick_check(py)
        if info and info.get("ultralytics") and info.get("torch"):
            return name, py
    return None
