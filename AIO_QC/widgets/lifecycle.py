# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
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
from AIO_QC.core.diag import ignored as _ignored

_STATE = {"reason": None, "code": None, "start": time.time(), "crash_file": None, "prev_crash": ""}
_ORPHANS: set = set()                   # threads detached from a closing window: kept alive until they finish


_RUNNING: set = set()                   # started threads: a strong reference until they finish


def retain(thread):
    """Keep `thread` (a QThread) alive until it finishes. Without this, dropping the last Python reference while
    it still runs — e.g. `self.worker = None` in a slot fed by the thread's own last signal — destroys a running
    QThread, which Qt answers with qFatal (abort)."""
    if thread in _RUNNING:
        return
    _RUNNING.add(thread)
    thread.finished.connect(lambda t=thread: _RUNNING.discard(t))


def detach(thread):
    """A running thread must never be deleted with the window or dialog it belongs to: Qt deletes a widget's
    children BEFORE it says it is going (so there is no moment to rescue them), and deleting a running QThread
    aborts the whole app ("QThread: Destroyed while thread … is still running"). So a thread is owned by Python
    from start() until it finishes: no Qt parent, a strong reference, its owner remembered for stop_threads()."""
    import weakref
    p = thread.parent()
    if p is not None:
        try:
            thread._aioqc_owner = weakref.ref(p)
        except TypeError:
            thread._aioqc_owner = lambda p=p: p
        thread.setParent(None)
    retain(thread)


def owner_of(thread):
    ref = getattr(thread, "_aioqc_owner", None)
    o = ref() if ref is not None else None
    try:
        from PyQt6 import sip
        if o is not None and sip.isdeleted(o):
            return None
    except ImportError:
        pass
    return o if o is not None else thread.parent()


class RetainedThread:
    """Mixin for EVERY QThread subclass of the app: start() names it (for the log), takes it out of its window's
    ownership and keeps it alive until it finishes (detach()); deleteLater() of a running one waits for its end."""

    def start(self, *a):
        if not self.objectName():
            self.setObjectName(type(self).__name__)
        detach(self)
        super().start(*a)

    def deleteLater(self):
        try:
            running = self.isRunning()
        except RuntimeError:
            return
        if running:                                  # never delete a running thread: try again shortly
            QTimer.singleShot(100, self.deleteLater)
            return
        super().deleteLater()


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
    crash = Path(log_dir) / "aio-qc-crash.log"
    try:
        if crash.is_file() and crash.stat().st_size > 0:
            logger.warning(f"The previous run ended with a native crash — see {crash}")
            try:
                _STATE["prev_crash"] = crash.read_text(encoding="utf-8", errors="replace")[:20000]
            except OSError:
                pass
            crash.rename(crash.with_name(f"aio-qc-crash-{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"))
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
    except Exception as _exc:
        _ignored(_exc)


def previous_crash() -> str:
    """The native-crash dump of the previous run ("" when it ended normally)."""
    return _STATE.get("prev_crash") or ""


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
    for t in list(_RUNNING):                         # threads detached at start() that belong to this owner
        try:
            if t in running or not t.isRunning():
                continue
        except RuntimeError:
            continue
        o = owner_of(t)
        while o is not None and o is not owner:
            try:
                o = o.parent()
            except RuntimeError:
                o = None
        if o is owner:
            running.append(t)
    for t in running:
        name = thread_name(t)
        logger.debug(f"stopping {name}" + (f" ({why})" if why else ""))
        for attr in ("stop", "cancel"):
            fn = getattr(t, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception as _exc:
                    _ignored(_exc)
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
            if t.parent() is not None:
                t.setParent(None)
            _ORPHANS.add(t)
            t.finished.connect(lambda t=t: (_ORPHANS.discard(t), logger.debug(f"{thread_name(t)} finished (detached)")))
    return len(running)


def stop_everything(window, why: str, timeout_ms: int = 4000):
    """App shutdown: every thread under the main window, detached stragglers and the thread pool."""
    n = stop_threads(window, why, timeout_ms)
    rest = []
    for t in list(_RUNNING):                         # anything else still going (any window, any mode)
        try:
            if t.isRunning():
                rest.append(t)
        except RuntimeError:
            pass
    for t in rest:
        for attr in ("stop", "cancel"):
            fn = getattr(t, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception as _exc:
                    _ignored(_exc)
        t.requestInterruption()
        t.quit()
    for t in rest:
        if not t.wait(timeout_ms):
            _ORPHANS.add(t)
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


def still_running() -> list:
    out = []
    for t in list(_RUNNING | _ORPHANS):
        try:
            if t.isRunning():
                out.append(t)
        except RuntimeError:
            pass
    return out


def final_exit(code: int):
    """The very end: a thread that is STILL running (stuck in a slow network read, say) would be destroyed by
    Python's own shutdown — an abort and a crash report for a normal exit. Then leave at once, log written."""
    left = still_running()
    if not left:
        return
    logger.warning(f"{len(left)} background thread(s) still busy at exit ({', '.join(thread_name(t) for t in left)})"
                   " — leaving without waiting for them")
    _at_exit()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as _exc:
        _ignored(_exc)
    os._exit(code if isinstance(code, int) else 0)
