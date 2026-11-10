# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""How much memory the computer has left (psutil when installed; otherwise unknown and every check passes)."""
from __future__ import annotations

LOW_BYTES = 700 * 1024 * 1024          # below this much free memory the app stops caching / preloading images


def _vm():
    try:
        import psutil
        return psutil.virtual_memory()
    except Exception:
        return None


def available() -> int | None:
    vm = _vm()
    return int(vm.available) if vm is not None else None


def total() -> int | None:
    vm = _vm()
    return int(vm.total) if vm is not None else None


def is_low(margin: int = LOW_BYTES) -> bool:
    a = available()
    return a is not None and a < margin


def cache_budget(ceiling: int, share: float = 0.12, floor: int = 128 * 1024 * 1024) -> int:
    """A cache size that fits this computer: `share` of its memory, between `floor` and `ceiling`."""
    t = total()
    if t is None:
        return ceiling
    return int(max(floor, min(ceiling, t * share)))


def human(n: int | None) -> str:
    if n is None:
        return "unknown"
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


def is_oom(exc: BaseException | None) -> bool:
    """Out of memory, from Python, numpy (_ArrayMemoryError), Qt (null image) or OpenCV ("Insufficient memory") —
    also when it is the cause of another error: OpenCV's binding turns a numpy allocation failure into
    "SystemError: <built-in function resize> returned a result with an exception set"."""
    seen = 0
    while exc is not None and seen < 8:
        if isinstance(exc, MemoryError):
            return True
        t = str(exc)
        if type(exc).__name__ == "error" and ("Insufficient memory" in t or "OutOfMemory" in t or
                                              "Failed to allocate" in t):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


def release_everywhere():
    """Ask every mode window to give back what it can (caches) — after an out-of-memory error anywhere."""
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        sh = getattr(app, "shell", None)
        for w in (getattr(sh, "image", None), getattr(sh, "video", None), getattr(sh, "studio", None)):
            rel = getattr(w, "release_memory", None)
            if rel is not None:
                rel()
        from .io import BUDGET
        BUDGET.trim(0.5)
    except Exception:
        pass
