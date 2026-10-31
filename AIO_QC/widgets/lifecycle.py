# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
App lifecycle through loguru: every way AIO QC can end is logged, Qt's own messages reach the log, and background
threads are never destroyed while running.

    install(app, log_dir)              Qt message handler · Ctrl+C / SIGTERM · crash hooks · exit record
    stop_threads(owner, why)           ask the owner's QThreads to stop and wait; detach (never destroy) stragglers
    set_exit_reason("user closed the window")
"""
from __future__ import annotations

import atexit
import faulthandler
import os
import signal
import sys
import threading
import time
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QThread, QThreadPool, QTimer, QtMsgType, qInstallMessageHandler

_STATE = {"reason": None, "code": None, "start": time.time(), "crash_file": None}
_ORPHANS: set = set()                   # threads detached from a closing window: kept alive until they finish


def set_exit_reason(reason: str, override: bool = False):
    if override or _STATE["reason"] is None:
        _STATE["reason"] = reason


def set_exit_code(code):
    _STATE["code"] = code


# =============================================================================
# Qt messages -> loguru (with flood control)
# =============================================================================
_LEVEL = {QtMsgType.QtDebugMsg: "DEBUG", QtMsgType.QtInfoMsg: "INFO", QtMsgType.QtWarningMsg: "WARNING",
          QtMsgType.QtCriticalMsg: "ERROR", QtMsgType.QtFatalMsg: "CRITICAL"}
_flood: dict[str, list] = {}            # message -> [count since last log, last log time]
_IGNORED = ("Invalid nullptr in QObject::connect",)


def _qt_handler(mode, ctx, message):
    if any(message.startswith(x) for x in _IGNORED):
        return
    level = _LEVEL.get(mode, "WARNING")
    where = ""
    if ctx is not None and ctx.file:
        where = f" ({ctx.file}:{ctx.line})"
    now = time.monotonic()
    f = _flood.get(message)
    if f is not None and now - f[1] < 5.0:          # the same message within 5 s: count it, log a summary later
        f[0] += 1
        return
    if f is not None and f[0]:
        logger.opt(depth=1).log(level, f"[Qt] {message}{where} (repeated {f[0]} more times)")
    else:
        logger.opt(depth=1).log(level, f"[Qt] {message}{where}")
    _flood[message] = [0, now]
    if len(_flood) > 500:
        _flood.clear()
    if "QThread: Destroyed while thread" in message:
        logger.error("[Qt] a running background thread was destroyed — please report this with the log above")


# =============================================================================
# Signals: Ctrl+C (SIGINT), SIGTERM, Ctrl+Break (Windows)
# =============================================================================
def _install_signals(app):
    def handler(signum, _frame):
        name = {getattr(signal, "SIGINT", None): "Ctrl+C (SIGINT)", getattr(signal, "SIGTERM", None): "SIGTERM",
                getattr(signal, "SIGBREAK", None): "Ctrl+Break"}.get(signum, f"signal {signum}")
        if _STATE.get("signalled"):
            logger.warning(f"{name} again — exiting immediately")
            set_exit_reason(f"{name} (forced)", override=True)
            os._exit(130)
        _STATE["signalled"] = True
        set_exit_reason(f"{name} from the terminal")
        logger.warning(f"{name} received — closing AIO QC gracefully (press again to force)")
        win = getattr(app, "main_window", None)
        QTimer.singleShot(0, (lambda: win.request_quit(name)) if win is not None else app.quit)

    for s in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, s):
            try:
                signal.signal(getattr(signal, s), handler)
            except (ValueError, OSError):
                pass
    # Qt's event loop does not return to Python on its own: a tick lets Python run the signal handler
    tick = QTimer(app)
    tick.setInterval(250)
    tick.timeout.connect(lambda: None)
    tick.start()
    app._signal_tick = tick


# =============================================================================
# Crashes
# =============================================================================
def _install_crash_hooks(log_dir: Path):
    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).error(
            f"Unhandled exception in thread {getattr(args.thread, 'name', '?')}")

    threading.excepthook = thread_hook

    def unraisable(u):
        logger.opt(exception=(u.exc_type, u.exc_value, u.exc_traceback)).warning(
            f"Unraisable exception: {u.err_msg or ''} {u.object!r}"[:400])

    sys.unraisablehook = unraisable
    # native crashes (segfaults in C extensions / Qt): Python cannot log them, faulthandler writes the stack
    crash = Path(log_dir) / "aio_qc_crash.log"
    try:
        if crash.is_file() and crash.stat().st_size > 0:
            logger.warning(f"The previous run ended with a native crash — see {crash}")
            crash.rename(crash.with_name(f"aio_qc_crash_{time.strftime('%Y%m%d_%H%M%S')}.log"))
        fh = open(crash, "w", encoding="utf-8")
        faulthandler.enable(file=fh, all_threads=True)
        _STATE["crash_file"] = (crash, fh)
    except OSError as exc:
        logger.debug(f"faulthandler not enabled: {exc}")


def _at_exit():
    up = time.time() - _STATE["start"]
    reason = _STATE["reason"] or "normal exit"
    code = _STATE["code"]
    logger.info(f"AIO QC exited — {reason}" + (f", exit code {code}" if code is not None else "")
                + f", ran {up / 60:.1f} min")
    cf = _STATE.get("crash_file")
    if cf:                                   # clean exit: no crash -> remove the empty dump
        path, fh = cf
        try:
            faulthandler.disable()
            fh.close()
            if path.stat().st_size == 0:
                path.unlink()
        except OSError:
            pass
    try:
        logger.complete()
    except Exception:
        pass


def install(app, log_dir: str | Path = "."):
    qInstallMessageHandler(_qt_handler)
    _install_signals(app)
    _install_crash_hooks(Path(log_dir))
    atexit.register(_at_exit)


# =============================================================================
# Threads
# =============================================================================
def thread_name(t: QThread) -> str:
    n = t.objectName()
    if n:
        return n
    fn = getattr(t, "fn", None)
    if fn is not None:
        return f"{type(t).__name__}({getattr(fn, '__qualname__', getattr(fn, '__name__', 'fn'))})"
    return type(t).__name__


def stop_threads(owner, why: str = "", timeout_ms: int = 3000) -> int:
    """Stop the owner's running QThreads before the owner is destroyed. Never destroys a running thread: a thread that
    does not finish in time is detached and kept alive until it ends (logged). Returns how many were running."""
    running = [t for t in owner.findChildren(QThread) if t.isRunning()]
    for t in running:
        name = thread_name(t)
        logger.debug(f"stopping {name}" + (f" ({why})" if why else ""))
        for attr in ("stop", "cancel"):
            fn = getattr(t, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        ev = getattr(t, "cancel_event", None)
        if ev is not None:
            ev.set()
        t.requestInterruption()
        t.quit()
    deadline = time.monotonic() + timeout_ms / 1000
    for t in running:
        left = max(0, int((deadline - time.monotonic()) * 1000))
        if not t.wait(left):
            name = thread_name(t)
            logger.warning(f"{name} is still running after {timeout_ms} ms ({why or 'owner closing'}) — detached, "
                           "it will finish in the background")
            t.setParent(None)
            _ORPHANS.add(t)
            t.finished.connect(lambda t=t: (_ORPHANS.discard(t), logger.debug(f"{thread_name(t)} finished (detached)")))
    return len(running)


def stop_everything(window, why: str, timeout_ms: int = 4000):
    """App shutdown: every thread under the main window, detached stragglers and the thread pool."""
    n = stop_threads(window, why, timeout_ms)
    for t in list(_ORPHANS):
        if t.isRunning() and not t.wait(timeout_ms):
            logger.warning(f"{thread_name(t)} did not finish before exit")
    pool = QThreadPool.globalInstance()
    if pool.activeThreadCount():
        logger.debug(f"waiting for {pool.activeThreadCount()} pooled task(s)")
        pool.clear()
        pool.waitForDone(timeout_ms)
    if n:
        logger.debug(f"{n} background thread(s) stopped ({why})")
