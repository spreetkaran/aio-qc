# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
External annotation editors (labelme — primary, labelImg — secondary for YOLO detection).

Sources  : the labelme / labelImg source zips (as downloaded from GitHub). Imported once: the project root is
           located, trimmed to what is needed to run (no examples / tests / builds — ~4 MB instead of 35-105 MB),
           patched where required (labelImg) and stored with a manifest. New versions are added the same way
           (Editor ▸ Add source zip…), e.g. labelme-6_4_0.zip; several versions can coexist.
Running  : from source, inside a managed environment that has only the dependencies of that source:
           * labelme reads its version from package metadata (`importlib.metadata.version("labelme")`) and its
             build backend needs git tags (hatch-vcs), so `pip install <zip>` fails. A tiny launcher supplies
             the version and runs `labelme.__main__.main()` from the source folder.
           * labelImg 1.8.6 is run as `python labelImg.py` from its (patched) source folder.
Envs     : one per editor kind and Python version, in the user's data folder, shared by all datasets and
           reused across sessions (installs are hundreds of MB). An existing environment can be used instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .io import atomic_write_text

LABELME, LABELIMG = "labelme", "labelimg"
TITLES = {LABELME: "labelme", LABELIMG: "labelImg"}
MIN_PYTHON = {LABELME: (3, 10), LABELIMG: (3, 8)}
MANIFEST = "aioqc_source.json"
BUILTIN_SOURCES = Path(__file__).resolve().parent.parent / "editors" / "sources"


def editors_home() -> Path:
    """Per-user folder for editor environments and imported sources (shared by all datasets)."""
    from .appdirs import data_home                       # portable builds: next to the executable
    return data_home() / "editors"


def user_sources_dir() -> Path:
    return editors_home() / "sources"


# =============================================================================
# labelImg patches (exact-match; a source that does not match is rejected, never half-patched)
# =============================================================================
LABELIMG_PATCHES = [
    # (file, old, new, why)
    ("labelImg.py", "import distutils.spawn\n", "",
     "distutils was removed in Python 3.12 (the import is unused)"),
    ("labelImg.py", "bar.setValue(bar.value() + bar.singleStep() * units)",
     "bar.setValue(int(bar.value() + bar.singleStep() * units))", "Qt needs an int (Python 3.10+)"),
    ("labelImg.py", "        self.zoom_widget.setValue(value)\n", "        self.zoom_widget.setValue(int(value))\n",
     "Qt needs an int (Python 3.10+)"),
    ("labelImg.py", "h_bar.setValue(new_h_bar_value)", "h_bar.setValue(int(new_h_bar_value))", "Qt needs an int"),
    ("labelImg.py", "v_bar.setValue(new_v_bar_value)", "v_bar.setValue(int(new_v_bar_value))", "Qt needs an int"),
    ("libs/canvas.py", "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
     "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))", "Qt needs ints"),
    ("libs/canvas.py", "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
     "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))", "Qt needs ints"),
    ("libs/canvas.py", "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
     "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))", "Qt needs ints"),
    ("libs/shape.py", "painter.drawText(min_x, min_y, self.label)",
     "painter.drawText(int(min_x), int(min_y), self.label)", "Qt needs ints"),
    ("labelImg.py", "self.label_file_format = settings.get(SETTING_LABEL_FILE_FORMAT, LabelFileFormat.PASCAL_VOC)",
     "self.label_file_format = LabelFileFormat.YOLO  # AIO QC: always YOLO (default was Pascal VOC XML)",
     "save YOLO .txt, never Pascal VOC .xml"),
    ("labelImg.py", "self.auto_saving.setChecked(settings.get(SETTING_AUTO_SAVE, False))",
     "self.auto_saving.setChecked(True)  # AIO QC: save when moving to the next image",
     "auto-save on, like labelme --autosave"),
]


def apply_labelme_patches(root: Path, version: str) -> list[str]:
    """
    Fixes for running several labelme versions side by side (applied when a source zip is added; tolerant — a patch
    whose code is not found is simply not needed for that version):
      * own settings file per major version (~/.labelme5rc, ~/.labelme6rc …): versions read each other's
        ~/.labelmerc and warn about / reject keys they do not know;
      * `import imgviz.io` -> `import imgviz` (the submodule import fails with some imgviz releases).
    """
    major = version.split(".")[0]
    rc = f".labelme{major}rc"
    applied = []
    for rel in ("labelme/config/__init__.py", "labelme/app.py"):
        p = root / rel
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            if ".labelmerc" in text:
                p.write_text(text.replace(".labelmerc", rc), encoding="utf-8")
                applied.append(f"{rel}: settings file ~/{rc} (not the shared ~/.labelmerc)")
    p = root / "labelme" / "utils" / "__init__.py"
    if p.is_file():
        text = p.read_text(encoding="utf-8")
        if "import imgviz.io\n" in text:
            p.write_text(text.replace("import imgviz.io\n", "import imgviz\n"), encoding="utf-8")
            applied.append("labelme/utils/__init__.py: import imgviz (imgviz.io import fails with some releases)")
    return applied


def apply_labelimg_patches(root: Path) -> list[str]:
    """Patch a labelImg source tree in place. Returns the list of applied patches; raises if any fails."""
    applied, failures = [], []
    for rel, old, new, why in LABELIMG_PATCHES:
        p = root / rel
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
        done = (old not in text) if not new else (new in text)     # removal patches: the old code is gone
        if done and text:
            applied.append(f"{rel}: already patched ({why})")
            continue
        if text.count(old) != 1:
            failures.append(f"{rel}: expected code not found ({why})")
            continue
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        applied.append(f"{rel}: {why}")
    if failures:
        raise ValueError("This labelImg source cannot be patched safely:\n  " + "\n  ".join(failures))
    return applied


# =============================================================================
# Sources
# =============================================================================
@dataclass
class EditorSource:
    kind: str
    version: str
    root: Path                       # folder that contains labelme/ (labelme) or labelImg.py (labelImg)
    origin: str = "bundled"          # bundled | imported
    flags: list = field(default_factory=list)      # CLI flags the source supports (labelme)
    deps: list = field(default_factory=list)
    patches: list = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{TITLES[self.kind]} {self.version}"

    @property
    def embeds_image_data_by_default(self) -> bool:
        """labelme 5.x stores imageData unless --nodata; 6.x does not unless --with-image-data."""
        return self.kind == LABELME and "--with-image-data" not in self.flags

    def deps_hash(self) -> str:
        return hashlib.sha1("\n".join(sorted(self.deps)).encode()).hexdigest()[:12]

    def to_json(self) -> dict:
        return {"kind": self.kind, "version": self.version, "flags": self.flags, "deps": self.deps,
                "patches": self.patches}


VERSION_RE = re.compile(r"(labelme|labelimg)[-_ ]?v?(\d+)[._](\d+)[._](\d+)", re.I)


def _labelme_flags(main_py: Path) -> list[str]:
    text = main_py.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r'"(--[a-z][a-z0-9-]*)"', text)))


def _pyproject_deps(pyproject: Path) -> list[str]:
    text = pyproject.read_text(encoding="utf-8")
    try:
        import tomllib                                              # Python 3.11+
        return list(tomllib.loads(text).get("project", {}).get("dependencies", []))
    except ImportError:
        m = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def _find_root(extracted: Path) -> tuple[str, Path]:
    for p in [extracted] + sorted(extracted.rglob("*"), key=lambda q: len(q.parts)):
        if not p.is_dir() or len(p.relative_to(extracted).parts) > 3:
            continue
        if (p / "labelme" / "__main__.py").is_file() and (p / "pyproject.toml").is_file():
            return LABELME, p
        if (p / "labelImg.py").is_file() and (p / "libs").is_dir():
            return LABELIMG, p
    raise ValueError("Not a labelme or labelImg source archive (no labelme/__main__.py or labelImg.py found).")


def _copytree(src: Path, dst: Path):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git*"), dirs_exist_ok=True)


def import_source_zip(zip_path: str | Path, dest_base: Path | None = None) -> EditorSource:
    """Extract, validate, trim, patch and register a source zip. Replaces an existing copy of the same version."""
    zip_path = Path(zip_path)
    dest_base = dest_base or user_sources_dir()
    with tempfile.TemporaryDirectory(prefix="aioqc_src_") as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        kind, root = _find_root(tmp)
        m = VERSION_RE.search(zip_path.name) or VERSION_RE.search(root.name)
        version = ".".join(m.groups()[1:]) if m else None
        if kind == LABELIMG and version is None:
            init = (root / "libs" / "__init__.py").read_text(encoding="utf-8", errors="replace") \
                if (root / "libs" / "__init__.py").is_file() else ""
            vm = re.search(r"__version__\s*=\s*['\"]([\d.]+)['\"]", init)
            version = vm.group(1) if vm else None
        if version is None:
            raise ValueError(f"Cannot tell the {TITLES[kind]} version from '{zip_path.name}'. "
                             f"Name the file like {TITLES[kind]}-1_2_3.zip.")
        dest = dest_base / f"{kind}-{version}"
        staging = dest_base / f".{kind}-{version}.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        if kind == LABELME:
            _copytree(root / "labelme", staging / "labelme")
            for f in ("pyproject.toml", "LICENSE", "README.md"):
                if (root / f).is_file():
                    shutil.copy2(root / f, staging / f)
            patches = apply_labelme_patches(staging, version)
            src = EditorSource(kind, version, dest, "imported", flags=_labelme_flags(staging / "labelme" / "__main__.py"),
                               deps=_pyproject_deps(staging / "pyproject.toml"), patches=patches)
            if "def main" not in (staging / "labelme" / "__main__.py").read_text(encoding="utf-8", errors="replace"):
                raise ValueError("labelme/__main__.py has no main() — unsupported labelme source.")
        else:
            shutil.copy2(root / "labelImg.py", staging / "labelImg.py")
            for d in ("libs", "resources", "data"):
                if (root / d).is_dir():
                    _copytree(root / d, staging / d)
            for f in ("LICENSE", "README.rst", "__init__.py"):
                if (root / f).is_file():
                    shutil.copy2(root / f, staging / f)
            if not (staging / "libs" / "resources.py").is_file():
                raise ValueError("libs/resources.py is missing (it must be generated with pyrcc5 before packaging).")
            patches = apply_labelimg_patches(staging)
            src = EditorSource(kind, version, dest, "imported", deps=["pyqt5", "lxml"], patches=patches)
        atomic_write_text(staging / MANIFEST, json.dumps(src.to_json(), indent=1))
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
        src.root = dest
        return src


def _read_source(folder: Path, origin: str) -> EditorSource | None:
    try:
        m = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        return EditorSource(m["kind"], m["version"], folder, origin, m.get("flags", []), m.get("deps", []),
                            m.get("patches", []))
    except (OSError, ValueError, KeyError):
        return None


def _vkey(v: str):
    return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))


def list_sources(kind: str | None = None) -> list[EditorSource]:
    """Bundled + imported sources, newest first; an imported copy overrides a bundled one of the same version."""
    found: dict[tuple, EditorSource] = {}
    for base, origin in ((BUILTIN_SOURCES, "bundled"), (user_sources_dir(), "imported")):
        if base.is_dir():
            for d in sorted(base.iterdir()):
                s = _read_source(d, origin) if d.is_dir() and not d.name.startswith(".") else None
                if s and (kind is None or s.kind == kind):
                    found[(s.kind, s.version)] = s
    return sorted(found.values(), key=lambda s: (s.kind, _vkey(s.version)), reverse=True)


# =============================================================================
# Environments
# =============================================================================
def python_version(py: str | Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run([str(py), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        a, b = out.split(".")
        return int(a), int(b)
    except Exception:
        return None


def find_base_python(minimum: tuple[int, int]) -> tuple[str, tuple[int, int]] | None:
    """A Python >= minimum to create the editor environment from (this app's Python first)."""
    cands = []
    if not getattr(sys, "frozen", False):
        cands.append(sys.executable)
    try:
        from .env_manager import discover_envs
        cands += [py for _name, py in discover_envs()]
    except Exception:
        pass
    seen = set()
    for py in cands:
        key = os.path.normcase(os.path.abspath(py))
        if key in seen:
            continue
        seen.add(key)
        v = python_version(py)
        if v and v >= minimum:
            return py, v
    return None


def env_dir(kind: str, pyver: tuple[int, int]) -> Path:
    return editors_home() / "envs" / f"{kind}-py{pyver[0]}.{pyver[1]}"


def env_state(env: Path) -> dict:
    try:
        return json.loads((env / "aioqc_env.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_installed(env: Path, source: EditorSource, python: str):
    st = env_state(env)
    st.setdefault("deps", {})[source.deps_hash()] = source.title
    st["python"] = python
    atomic_write_text(env / "aioqc_env.json", json.dumps(st, indent=1))


def is_installed(env: Path, source: EditorSource, python: str | None = None) -> bool:
    py = python or _venv_python(env)
    return bool(py and Path(py).is_file() and source.deps_hash() in env_state(env).get("deps", {}))


def _venv_python(env: Path) -> str:
    return str(env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))


def install_steps(env: Path, source: EditorSource, base_python: str, create: bool) -> list[tuple[str, list[str]]]:
    """[(title, argv)] for the GUI runner: create venv (if needed), upgrade pip, install the source's deps."""
    py = _venv_python(env)
    pip = [py, "-m", "pip", "install", "--disable-pip-version-check"]
    steps = []
    if create:
        steps.append(("Create the editor environment", [base_python, "-m", "venv", str(env)]))
        steps.append(("Upgrade pip", pip + ["--upgrade", "pip"]))
    steps.append((f"Install {source.title} dependencies ({len(source.deps)} packages)", pip + list(source.deps)))
    steps.append(("Check the installation", [py, "-c", CHECK_SNIPPET[source.kind]]))
    return steps


CHECK_SNIPPET = {
    LABELME: "import PyQt5.QtWidgets, onnxruntime, osam, imgviz, yaml; print('labelme dependencies OK')",
    LABELIMG: "import PyQt5.QtWidgets, lxml; print('labelImg dependencies OK')",
}


def remove_editor_env(env: Path) -> bool:
    try:
        shutil.rmtree(env)
        return True
    except OSError:
        return False


# =============================================================================
# Launching
# =============================================================================
LAUNCHER = '''"""AIO QC launcher: run labelme from a source folder (no pip install / package metadata needed)."""
import importlib.metadata as _md
import sys

src, version = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
_orig = _md.version


def _version(name):
    if name.lower() == "labelme":
        return version
    return _orig(name)


_md.version = _version
sys.argv = ["labelme"] + sys.argv[3:]
from labelme.__main__ import main  # noqa: E402

sys.exit(main())
'''


def launcher_path() -> Path:
    p = editors_home() / "aioqc_labelme_launcher.py"
    if not p.is_file() or p.read_text(encoding="utf-8") != LAUNCHER:
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, LAUNCHER)
    return p


def _flag(source: EditorSource, *names: str) -> str | None:
    return next((n for n in names if n in source.flags), None)


def build_command(source: EditorSource, python: str, work_dir: Path, labels_file: Path | None,
                  image_data: str, validate: bool = True, flags_file: Path | None = None) -> tuple[str, list[str], str]:
    """(program, args, cwd). image_data: AIO QC's write-back policy ('embed' | 'none'); the editor itself always
    runs without image data.
    validate=False offers the labels as suggestions only (LabelMe datasets without a class list)."""
    if source.kind == LABELIMG:
        return python, [str(source.root / "labelImg.py"), str(work_dir), str(labels_file), str(work_dir)], \
            str(source.root)
    args = [str(launcher_path()), str(source.root), source.version, str(work_dir)]
    if labels_file is not None:
        args += [_flag(source, "--labels") or "--labels", str(labels_file)]
        v = _flag(source, "--validate-label", "--validatelabel") if validate else None
        if v:
            args += [v, "exact"]
        ns = _flag(source, "--no-sort-labels", "--nosortlabels")
        if ns:
            args.append(ns)
    if flags_file is not None and "--flags" in source.flags:     # classification: image flags = classes
        args += ["--flags", str(flags_file)]
    auto = _flag(source, "--autosave")
    if auto:
        args.append(auto)
    # The editor never embeds the image: its copies stay a few KB and AIO QC decides on write-back whether the
    # dataset JSON carries imageData (labelme 5 embeds by default -> --nodata; labelme 6 does not by default).
    if "--nodata" in source.flags:
        args.append("--nodata")
    return python, args, str(work_dir)
