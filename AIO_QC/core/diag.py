# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Errors that are handled by carrying on (a thumbnail that cannot be read, a cache that cannot be written …) still leave
a trace in the log, so a problem that repeats can be found:

    try:
        ...
    except Exception as exc:
        ignored(exc)                       # one DEBUG line: where, the error type and its text
        return None

Cheap enough for hot loops: per code location the first 3 are logged, then the 10th, 100th, 1000th and every
10,000th with the count — a million unreadable files give a handful of lines. No traceback and no variable values
(the log file keeps those for real crashes only), and anything that looks like a key or token is masked.
"""
from __future__ import annotations

import re
import sys

from loguru import logger

_COUNTS: dict = {}
_TYPES: dict = {}                 # (file, line) -> {exception type: count} — what each handler really catches
_SECRET = re.compile(r"(?i)(sk-[a-z0-9_\-]{8,}|(?:api[_-]?key|token|secret|password|authorization)"
                     r"[\"']?\s*[:=]\s*[\"']?[^\s\"'&,;]+|bearer\s+[a-z0-9._\-]{8,})")


def _mask(text: str) -> str:
    return _SECRET.sub(lambda m: m.group(0)[:6] + "…[masked]", text)


def ignored(exc: BaseException, what: str = "") -> None:
    """Note an error that the caller handles by carrying on."""
    try:
        f = sys._getframe(1)
        key = (f.f_code.co_filename, f.f_lineno)
    except Exception:
        key = ("?", 0)
    n = _COUNTS.get(key, 0) + 1
    _COUNTS[key] = n
    seen = _TYPES.setdefault(key, {})
    t = type(exc).__name__
    seen[t] = seen.get(t, 0) + 1
    if n > 3 and n not in (10, 100, 1000) and n % 10000:
        return
    try:
        text = _mask(str(exc))[:300]
        logger.opt(depth=1).debug("ignored{} {}: {}{}", f" ({what})" if what else "", type(exc).__name__, text,
                                  f"  [×{n} here]" if n > 3 else "")
    except Exception:
        pass                                            # logging must never be the thing that fails


def counts() -> dict:
    """{(file, line): how many times} — for tests and diagnostics."""
    return dict(_COUNTS)


def types_seen() -> dict:
    """{"file:line": {exception type: count}} — which errors each handler has actually caught in this process.
    The evidence for narrowing a handler to the exceptions it is really for (AIOQC_DIAG_DUMP writes it at the end
    of a test run)."""
    return {f"{f}:{ln}": dict(v) for (f, ln), v in _TYPES.items()}


def report() -> None:
    """At exit: one DEBUG line per handler that caught something this session (busiest first, at most 40), with the
    error types — so everyday use shows which handlers are busy and what they really catch. AIOQC_DIAG_DUMP=<file>
    also writes it as JSON."""
    try:
        rows = sorted(_TYPES.items(), key=lambda kv: -sum(kv[1].values()))
        if rows:
            logger.debug("handled errors this session: {} place(s)", len(rows))
        for (f, ln), v in rows[:40]:
            logger.debug("  {}:{}  {}", f, ln, ", ".join(f"{k} ×{n}" for k, n in sorted(v.items(), key=lambda x: -x[1])))
        import os
        path = os.environ.get("AIOQC_DIAG_DUMP")
        if path:
            import json
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(types_seen(), fh, indent=1)
    except Exception:
        pass                                            # never the reason an exit fails
