# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
API keys live in environment variables — never in a config, a log or a report (Qt-free).

    get(name)            the key, from this process's environment; else from where save() put it (an app started
                         before the key was saved still finds it)
    save(name, value)    sets it for this process AND for later ones:
                           Windows       a user environment variable (HKCU\\Environment), announced to running apps
                           macOS/Linux   an `export` line in the shell profile(s), in a block marked "AIO QC"
    forget(name)         removes what save() wrote
    mask(value)          "sk-…9Qg" for showing that a key is there
"""
from __future__ import annotations

import os
import re
import sys
from AIO_QC.core.diag import ignored as _ignored

MARK = "# >>> AIO QC key {name} >>>"
END = "# <<< AIO QC key {name} <<<"


def valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


def mask(value: str) -> str:
    if not value:
        return ""
    v = str(value)
    return (v[:3] + "…" + v[-3:]) if len(v) > 8 else "•" * len(v)


# ------------------------------------------------------------------------------------------------ windows
def _win_get(name):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0])
    except (OSError, ImportError):
        return ""


def _win_broadcast():
    try:
        import ctypes
        HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x1A, 0x0002
        res = ctypes.c_ulong()
        ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",
                                                 SMTO_ABORTIFHUNG, 3000, ctypes.byref(res))
    except Exception as _exc:
        _ignored(_exc)


def _win_set(name, value):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
        if value is None:
            try:
                winreg.DeleteValue(k, name)
            except FileNotFoundError:
                pass
        else:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
    _win_broadcast()


# ------------------------------------------------------------------------------------------------ posix
def profiles(home: str | None = None) -> list[str]:
    """Shell start-up files that get the export line (those that exist, plus ~/.profile)."""
    home = home or os.path.expanduser("~")
    names = [".profile", ".bashrc", ".zshrc", ".zprofile", ".bash_profile"]
    out = [os.path.join(home, n) for n in names if os.path.exists(os.path.join(home, n))]
    if os.path.join(home, ".profile") not in out:
        out.insert(0, os.path.join(home, ".profile"))
    return out


def _quote(v):
    return "'" + v.replace("'", "'\"'\"'") + "'"


def _strip_block(text, name):
    pat = re.compile(rf"\n?{re.escape(MARK.format(name=name))}\n.*?{re.escape(END.format(name=name))}\n?", re.S)
    return pat.sub("\n", text)


def _posix_get(name, home=None):
    pat = re.compile(rf"{re.escape(MARK.format(name=name))}\nexport {re.escape(name)}=(.*)\n")
    for p in profiles(home):
        try:
            with open(p, encoding="utf-8") as f:
                m = pat.search(f.read())
        except OSError:
            continue
        if m:
            raw = m.group(1).strip()
            if raw.startswith("'") and raw.endswith("'"):
                return raw[1:-1].replace("'\"'\"'", "'")
            return raw
    return ""


def _posix_set(name, value, home=None) -> list[str]:
    written = []
    for p in profiles(home):
        try:
            text = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        except OSError:
            continue
        text = _strip_block(text, name).rstrip("\n")
        if value is not None:
            text += f"\n\n{MARK.format(name=name)}\nexport {name}={_quote(value)}\n{END.format(name=name)}"
        text += "\n"
        tmp = p + ".aioqc.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text.lstrip("\n") if not text.strip() else text)
        try:
            os.chmod(tmp, os.stat(p).st_mode if os.path.exists(p) else 0o600)
        except OSError:
            pass
        os.replace(tmp, p)
        written.append(p)
    return written


# ------------------------------------------------------------------------------------------------ api
def get(name: str, home: str | None = None) -> str:
    if not name:
        return ""
    v = os.environ.get(name, "")
    if v:
        return v
    v = _win_get(name) if sys.platform == "win32" else _posix_get(name, home)
    if v:
        os.environ[name] = v
    return v


def save(name: str, value: str, persist: bool = True, home: str | None = None) -> str:
    """Store the key; returns where it was saved (for the message the app shows)."""
    if not valid_name(name):
        raise ValueError(f"“{name}” is not a valid environment variable name (letters, digits and _).")
    value = (value or "").strip()
    if not value:
        raise ValueError("The key is empty.")
    os.environ[name] = value
    if not persist:
        return "this session only"
    if sys.platform == "win32" and home is None:
        _win_set(name, value)
        return "your Windows user environment variables"
    files = _posix_set(name, value, home)
    return "your shell profile (" + ", ".join(os.path.basename(f) for f in files) + ")"


def forget(name: str, home: str | None = None):
    os.environ.pop(name, None)
    if sys.platform == "win32" and home is None:
        try:
            _win_set(name, None)
        except OSError:
            pass
    else:
        _posix_set(name, None, home)
