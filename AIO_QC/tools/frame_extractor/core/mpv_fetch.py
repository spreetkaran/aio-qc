# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Get libmpv for this computer automatically - Qt-free (runs in a worker thread; progress / cancel are callbacks).

    plan()                       what would be done here: {"how": "download" | "package" | None, ...}
    fetch(dest, progress, cancel) does it; returns the libmpv path (download) or "" (system package installed)

Per OS - the same sources packaging/LIBMPV.md explains:
  Windows  the latest zhongfly/mpv-winbuild release, "mpv-dev-lgpl-<arch>-*.7z" (LGPL-2.1+). Only libmpv-2.dll is
           taken (statically linked, ~30 MB). Release list from GitHub's API, or its release page when the API is
           rate-limited; SHA-256 checked when GitHub publishes it. Extracted with py7zr, Windows' own tar.exe
           (libarchive reads 7z) or 7-Zip.
  macOS    conda-forge's mpv (relocatable, built for old macOS versions, both architectures), installed into the
           per-user libmpv folder with micromamba - itself fetched from micro.mamba.pm or conda-forge, checksum
           verified. Every package micromamba installs is checksum-verified by it. Larger (~250-400 MB with ffmpeg).
  Linux    the distribution's package (libmpv2 / libmpv1 / mpv-libs / mpv) through pkexec (the system asks for the
           password). A libmpv downloaded from elsewhere is built against other system libraries and fails inside
           the app ("undefined symbol" against the GLib / fontconfig Qt already loaded) - tested - so Linux uses
           the package manager, never a download.
Everything lands in the per-user libmpv folder (mpv_lib.user_lib_dir()), which the loader searches; nothing is
written next to the app (it may be read-only). A file in use is replaced at the next start (<name>.new).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

WIN_REPO = "zhongfly/mpv-winbuild"
CONDA = "https://conda.anaconda.org/conda-forge"
UA = {"User-Agent": "AIO-QC-libmpv-fetch"}


class Cancelled(Exception):
    pass


def _noop(*_a, **_k):
    return False


def machine() -> str:
    m = platform.machine().lower()
    return "aarch64" if m in ("arm64", "aarch64") else "x86_64"


def conda_subdir(plat: str = sys.platform, arch: str | None = None) -> str:
    arch = arch or machine()
    if plat == "darwin":
        return "osx-arm64" if arch == "aarch64" else "osx-64"
    return "linux-aarch64" if arch == "aarch64" else "linux-64"


# ------------------------------------------------------------------ what to do here
def linux_package_command(os_release: str | None = None, which=shutil.which) -> str | None:
    """Shell command installing libmpv with this system's package manager (None: no known one)."""
    if which("apt-get"):
        return ("apt-get install -y libmpv2 || apt-get install -y libmpv1")
    if which("dnf"):
        return "dnf install -y mpv-libs"
    if which("pacman"):
        return "pacman -S --noconfirm --needed mpv"
    if which("zypper"):
        return "zypper --non-interactive install libmpv2"
    if which("apk"):
        return "apk add mpv-libs"
    return None


def plan(plat: str = sys.platform, which=shutil.which) -> dict:
    if plat == "win32":
        return {"how": "download", "auto": True, "what": "libmpv-2.dll (LGPL build, about 30 MB)",
                "source": f"github.com/{WIN_REPO}"}
    if plat == "darwin":
        return {"how": "download", "auto": True, "what": "mpv from conda-forge (about 300 MB with its libraries)",
                "source": "conda-forge"}
    cmd = linux_package_command(which=which)
    if cmd is None:
        return {"how": None, "auto": False, "what": "", "source": "",
                "why": "no known package manager - install your distribution's libmpv package"}
    root = hasattr(os, "geteuid") and os.geteuid() == 0
    runner = [] if root else (["pkexec"] if which("pkexec") else None)
    if runner is None:
        return {"how": None, "auto": False, "what": "", "source": "", "command": "sudo sh -c '" + cmd + "'",
                "why": "pkexec is not available - run the command in a terminal"}
    return {"how": "package", "auto": False, "what": "libmpv from your distribution (asks for your password)",
            "source": "package manager", "argv": runner + ["sh", "-c", cmd], "command": "sudo sh -c '" + cmd + "'"}


# ------------------------------------------------------------------ HTTP
def _open(url, token=None, timeout=60):
    h = dict(UA)
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def get_text(url, token=None) -> tuple[str, str]:
    """(body, final URL after redirects)."""
    with _open(url, token) as r:
        return r.read().decode("utf-8", "replace"), r.geturl()


def download(url, dst: Path, progress=_noop, cancel=_noop, label="", sha256=None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    with _open(url, timeout=120) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, t_last = 0, 0.0
        while True:
            if cancel():
                raise Cancelled()
            b = r.read(1 << 18)
            if not b:
                break
            f.write(b)
            h.update(b)
            got += len(b)
            now = time.monotonic()
            if now - t_last > 0.2:
                t_last = now
                progress(got / total if total else None, f"{label} {got / 1e6:.0f}" +
                         (f" / {total / 1e6:.0f} MB" if total else " MB"))
    digest = h.hexdigest()
    if sha256 and digest.lower() != sha256.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dst.name} (expected {sha256[:12]}…, got {digest[:12]}…)")
    os.replace(part, dst)
    return dst


# ------------------------------------------------------------------ Windows
def choose_windows_asset(release: dict, arch: str, gpl: bool = False, v3: bool = False) -> dict | None:
    """The mpv-dev archive for arch from a GitHub release JSON (pure).
    LGPL: mpv-dev-lgpl-<arch>-*.7z   GPL: mpv-dev-<arch>-*.7z (x86_64-v3 = AVX2 builds, only with v3)."""
    kind = ("mpv-dev-" if gpl else "mpv-dev-lgpl-") + arch + "-"
    want_v3 = bool(v3 and arch == "x86_64")
    by_v3: dict[bool, list[dict]] = {True: [], False: []}
    for a in release.get("assets", []):
        n = a.get("name", "")
        if n.startswith(kind) and n.endswith(".7z"):
            by_v3[n[len(kind):].startswith("v3-")].append(a)
    pick = by_v3[want_v3] or by_v3[False]
    return sorted(pick, key=lambda a: a["name"])[-1] if pick else None


def parse_release_page(html: str, repo: str, tag: str) -> dict:
    """A release JSON-alike from GitHub's expanded_assets HTML (used when the API is rate-limited)."""
    assets, seen = [], set()
    pat = re.compile(r'href="(/' + re.escape(repo) + r'/releases/download/' + re.escape(tag) + r'/([^"/]+))"')
    for m in pat.finditer(html):
        if m.group(2) not in seen:
            seen.add(m.group(2))
            assets.append({"name": m.group(2), "browser_download_url": "https://github.com" + m.group(1)})
    return {"tag_name": tag, "assets": assets}


def windows_release(repo=WIN_REPO, token=None) -> dict:
    try:
        body, _ = get_text(f"https://api.github.com/repos/{repo}/releases/latest", token)
        rel = json.loads(body)
        if rel.get("assets"):
            return rel
    except Exception:
        pass                                    # rate limit / blocked API: read the release page instead
    _, final = get_text(f"https://github.com/{repo}/releases/latest")
    tag = final.rstrip("/").rsplit("/", 1)[-1]
    html, _ = get_text(f"https://github.com/{repo}/releases/expanded_assets/{tag}")
    return parse_release_page(html, repo, tag)


def extract_dll(archive: Path, out: Path, name="libmpv-2.dll") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            want = [n for n in z.getnames() if n.replace("\\", "/").rsplit("/", 1)[-1].lower() == name]
            if want:
                z.extract(path=out, targets=want)
                return out / want[0]
            errors.append("py7zr: not in the archive")
    except ImportError:
        errors.append("py7zr not installed")
    except Exception as e:
        errors.append(f"py7zr: {e}")
    for exe in ([os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")]
                if sys.platform == "win32" else []) + [shutil.which("bsdtar"), shutil.which("7z"),
                                                        shutil.which("7za")]:
        if not exe or not os.path.isfile(exe):
            continue
        args = ([exe, "-xf", str(archive), "-C", str(out)] if "tar" in os.path.basename(exe).lower()
                else [exe, "e", "-y", f"-o{out}", str(archive), name, "-r"])
        r = subprocess.run(args, capture_output=True, text=True, **_no_window())
        hit = next(iter(out.rglob(name)), None)
        if hit is not None:
            return hit
        errors.append(f"{os.path.basename(exe)}: {r.stderr.strip()[:200]}")
    raise RuntimeError("could not extract libmpv-2.dll (" + "; ".join(errors) + ") - pip install py7zr")


def _no_window():
    return {"creationflags": 0x08000000} if sys.platform == "win32" else {}     # CREATE_NO_WINDOW


def fetch_windows(dest: Path, progress=_noop, cancel=_noop, gpl=False, token=None) -> Path:
    progress(None, "Finding the latest libmpv build…")
    rel = windows_release(token=token)
    asset = choose_windows_asset(rel, machine(), gpl)
    if asset is None:
        raise RuntimeError(f"no mpv-dev archive for {machine()} in {WIN_REPO} {rel.get('tag_name', '')}")
    digest = (asset.get("digest") or "").split(":", 1)[-1] if str(asset.get("digest", "")).startswith("sha256:") \
        else None
    work = dest / ".download"
    shutil.rmtree(work, ignore_errors=True)
    arc = download(asset["browser_download_url"], work / asset["name"], progress, cancel, "libmpv", digest)
    if cancel():
        raise Cancelled()
    progress(None, "Extracting libmpv-2.dll…")
    dll = extract_dll(arc, work / "x")
    target = dest / "libmpv-2.dll"
    shutil.copy2(dll, target.with_name(target.name + ".new"))
    try:
        os.replace(target.with_name(target.name + ".new"), target)
    except OSError:
        pass                                    # in use: mpv_lib promotes the .new copy at the next start
    (dest / "SOURCE.txt").write_text(
        f"libmpv-2.dll from https://github.com/{WIN_REPO}/releases/tag/{rel.get('tag_name', '')}\n"
        f"archive: {asset['name']}\nlicence: {'GPL-2.0-or-later' if gpl else 'LGPL-2.1-or-later'}\n"
        f"downloaded: {time.strftime('%Y-%m-%d %H:%M')}\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return target


# ------------------------------------------------------------------ macOS (conda-forge)
def _extract_micromamba(tb: Path, out: Path) -> Path:
    with tarfile.open(tb) as t:
        m = t.getmember("bin/micromamba")
        m.name = "micromamba"
        if hasattr(tarfile, "data_filter"):
            t.extract(m, out, filter="data")
        else:
            t.extract(m, out)
    exe = out / "micromamba"
    exe.chmod(0o755)
    return exe


def ensure_micromamba(dest: Path, subdir: str, progress=_noop, cancel=_noop) -> Path:
    exe = dest / "micromamba"
    if exe.is_file() and os.access(exe, os.X_OK):
        return exe
    tmp = dest / ".download"
    tmp.mkdir(parents=True, exist_ok=True)
    errors = []
    try:                                                    # 1) the official installer endpoint
        tb = download(f"https://micro.mamba.pm/api/micromamba/{subdir}/latest", tmp / "micromamba.tar.bz2",
                      progress, cancel, "micromamba")
        return _extract_micromamba(tb, dest)
    except Cancelled:
        raise
    except Exception as e:
        errors.append(f"micro.mamba.pm: {e}")
    try:                                                    # 2) anaconda.org's file list (with checksums)
        body, _ = get_text("https://api.anaconda.org/package/conda-forge/micromamba/files")
        files = [f for f in json.loads(body) if f.get("attrs", {}).get("subdir") == subdir
                 and f.get("basename", "").endswith(".tar.bz2")]
        files.sort(key=lambda f: f.get("upload_time", ""))
        f = files[-1]
        url = f["download_url"] if f["download_url"].startswith("http") else "https:" + f["download_url"]
        tb = download(url, tmp / "micromamba.tar.bz2", progress, cancel, "micromamba", f.get("sha256"))
        return _extract_micromamba(tb, dest)
    except Cancelled:
        raise
    except Exception as e:
        errors.append(f"anaconda.org: {e}")
    try:                                                    # 3) conda-forge's index (large, but always there)
        progress(None, "Reading the conda-forge index…")
        body, _ = get_text(f"{CONDA}/{subdir}/current_repodata.json")
        pk = [(v["version"], k, v.get("sha256")) for k, v in json.loads(body).get("packages", {}).items()
              if v.get("name") == "micromamba"]
        pk.sort(key=lambda x: [int(p) if p.isdigit() else 0 for p in x[0].split(".")])
        _ver, fn, sha = pk[-1]
        tb = download(f"{CONDA}/{subdir}/{fn}", tmp / fn, progress, cancel, "micromamba", sha)
        return _extract_micromamba(tb, dest)
    except Cancelled:
        raise
    except Exception as e:
        errors.append(f"conda-forge: {e}")
    raise RuntimeError("could not get micromamba - " + "; ".join(errors))


def run_streaming(argv, progress=_noop, cancel=_noop, env=None, label="") -> tuple[int, str]:
    """Run a process, pass its output lines to progress, stop it on cancel. (returncode, last lines)."""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                         errors="replace", **_no_window())
    tail = []
    import threading
    stop = threading.Event()

    def watch():
        while not stop.wait(0.3):
            if cancel():
                try:
                    p.terminate()
                except OSError:
                    pass
                return
    threading.Thread(target=watch, daemon=True).start()
    try:
        for line in p.stdout:
            line = line.strip()
            if line:
                tail = (tail + [line])[-12:]
                progress(None, f"{label}{line[:120]}")
        rc = p.wait()
    finally:
        stop.set()
    if cancel():
        raise Cancelled()
    return rc, "\n".join(tail)


def fetch_conda(dest: Path, progress=_noop, cancel=_noop, subdir: str | None = None, spec="mpv") -> Path:
    subdir = subdir or conda_subdir()
    mm = ensure_micromamba(dest, subdir, progress, cancel)
    env_new, env = dest / "env.new", dest / "env"
    shutil.rmtree(env_new, ignore_errors=True)
    root = dest / "mamba-root"
    e = dict(os.environ, MAMBA_ROOT_PREFIX=str(root), CONDA_PKGS_DIRS=str(root / "pkgs"),
             CONDA_SUBDIR=subdir, MAMBA_NO_BANNER="1")
    progress(None, "Installing mpv from conda-forge…")
    rc, tail = run_streaming([str(mm), "create", "-y", "-p", str(env_new), "-c", "conda-forge",
                              "--override-channels", "--platform", subdir, spec], progress, cancel, e, "mpv: ")
    if rc != 0:
        shutil.rmtree(env_new, ignore_errors=True)
        raise RuntimeError(f"micromamba failed ({rc}): {tail[-400:]}")
    name = "libmpv.2.dylib" if subdir.startswith("osx") else "libmpv.so.2"
    if not (env_new / "lib" / name).is_file():
        raise RuntimeError(f"the installed mpv has no lib/{name}")
    prune_env(env_new)
    if env.exists():
        try:
            shutil.rmtree(env)
        except OSError:                                     # in use: keep using it until the next start
            pass
    if not env.exists():
        os.replace(env_new, env)
    shutil.rmtree(root / "pkgs", ignore_errors=True)        # the package cache: hundreds of MB, not needed
    shutil.rmtree(dest / ".download", ignore_errors=True)
    (dest / "SOURCE.txt").write_text(f"mpv from conda-forge ({subdir}), installed with micromamba into env/\n"
                                     f"licence: GPL-2.0-or-later\ndownloaded: {time.strftime('%Y-%m-%d %H:%M')}\n",
                                     encoding="utf-8")
    return (env if env.exists() else env_new) / "lib" / name


def prune_env(env: Path) -> int:
    """Drop what libmpv never loads: programs, headers, docs, static libraries, Python's standard library (only
    mpv's optional VapourSynth filter would start Python), OpenVINO plugins (an ffmpeg filter). Roughly halves the
    download's footprint. Every shared library stays, so the dynamic loader still resolves libmpv completely."""
    freed = 0

    def rm(p: Path):
        nonlocal freed
        if p.is_symlink() or p.is_file():
            freed += p.lstat().st_size
            p.unlink()
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and not f.is_symlink():
                    freed += f.stat().st_size
            shutil.rmtree(p, ignore_errors=True)
    for name in ("bin", "include", "man", "doc"):
        rm(env / name)
    for name in ("doc", "man", "info", "locale", "gtk-doc", "aclocal", "bash-completion", "zsh", "vala",
                 "gir-1.0", "applications", "icons"):
        rm(env / "share" / name)
    lib = env / "lib"
    for p in list(lib.glob("python3*")) + list(lib.glob("openvino-*")) + [lib / "pkgconfig", lib / "cmake"]:
        rm(p)
    for p in lib.glob("*.a"):
        rm(p)
    return freed


# ------------------------------------------------------------------ Linux (package manager)
def install_package(argv, progress=_noop, cancel=_noop) -> str:
    progress(None, "Installing libmpv with your package manager…")
    rc, tail = run_streaming(argv, progress, cancel, None, "")
    if rc in (126, 127) and argv and argv[0] == "pkexec":
        raise RuntimeError("the password prompt was cancelled or pkexec is not allowed here")
    if rc != 0:
        raise RuntimeError(f"the package manager failed ({rc}): {tail[-400:]}")
    return ""


# ------------------------------------------------------------------ one entry point
def fetch(dest, progress=_noop, cancel=_noop, plat: str = sys.platform) -> str:
    """Do what plan() says. Returns the downloaded libmpv path, or "" after a system package install."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    p = plan(plat)
    if p["how"] == "download" and plat == "win32":
        return str(fetch_windows(dest, progress, cancel, token=os.environ.get("GITHUB_TOKEN") or None))
    if p["how"] == "download":
        return str(fetch_conda(dest, progress, cancel))
    if p["how"] == "package":
        return install_package(p["argv"], progress, cancel)
    raise RuntimeError(p.get("why", "no automatic way to get libmpv here"))
