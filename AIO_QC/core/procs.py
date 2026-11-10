# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Safe child processes (Qt-free).

Every long-lived process AIO QC starts (the inference servers, and the Frame Extractor's ffmpeg pipes / samplers /
benchmarks, batch workers and their Manager) goes through here, so that:

  * it can never outlive the app - not even after a crash or a kill:
      Windows : the app puts itself in a Job Object with KILL_ON_JOB_CLOSE; every descendant inherits it and the
                OS kills them all when the app's process ends, however it ends.
      Linux   : children get PR_SET_PDEATHSIG = SIGKILL. That signal follows the *thread* that started the child,
                so all children are started from one long-lived launcher thread. It is set by `setpriv` (util-linux)
                exec'ing the command - never by Python code in the forked child (preexec_fn), which can deadlock
                when another thread held a lock (allocator, logging, ...) at the moment of the fork.
      macOS   : explicit cleanup (kill_all at exit).
  * background work (analysis, batch extraction) runs at below-normal priority, so the UI and the user's other
    programs stay responsive;
  * nothing pops up a console window on Windows;
  * kill_all() at exit stops whatever is still running.
"""
import atexit
import math
import os
import queue
import subprocess
import sys
import threading
import time
import weakref

WIN, LINUX = sys.platform == "win32", sys.platform.startswith("linux")
CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_live = weakref.WeakSet()
_job = None

_prctl = None
if LINUX:
    try:
        import ctypes
        _prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl
    except Exception:
        _prctl = None
PR_SET_PDEATHSIG, SIGKILL = 1, 9


def _preexec(nice):
    def hook():
        if _prctl is not None:
            _prctl(PR_SET_PDEATHSIG, SIGKILL)
        if nice:
            try:
                os.nice(nice)
            except OSError:
                pass
    return hook


class _Launcher:
    """All Popen calls happen on this one long-lived thread (PDEATHSIG follows the creating thread)."""

    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, name="process-launcher", daemon=True).start()

    def _run(self):
        while True:
            args, kw, box = self.q.get()
            try:
                box["p"] = subprocess.Popen(*args, **kw)
            except BaseException as e:
                box["e"] = e
            box["ev"].set()

    def popen(self, *args, **kw):
        box = {"ev": threading.Event()}
        self.q.put((args, kw, box))
        box["ev"].wait()
        if "e" in box:
            raise box["e"]
        return box["p"]


_launcher = None                    # started on the first child process (importing this module costs nothing)
_launcher_lock = threading.Lock()


def _get_launcher():
    global _launcher
    with _launcher_lock:
        if _launcher is None:
            _launcher = _Launcher()
        return _launcher


_wrap = None


def _linux_prefix(background):
    """Command prefix applying PDEATHSIG (setpriv) and low priority (nice) at exec time - no Python in the child."""
    global _wrap
    if _wrap is None:
        import shutil
        _wrap = {}
        sp = shutil.which("setpriv")
        if sp:
            try:
                h = subprocess.run([sp, "--help"], capture_output=True, text=True, timeout=5)
                if "pdeathsig" in (h.stdout + h.stderr):
                    _wrap["setpriv"] = [sp, "--pdeathsig", "KILL", "--"]
            except Exception:
                pass
        nc = shutil.which("nice")
        if nc:
            _wrap["nice"] = [nc, "-n", "10"]
    return _wrap.get("setpriv", []) + (_wrap.get("nice", []) if background else [])


def popen(cmd, background=False, **kw):
    """subprocess.Popen that can't outlive the app. background=True: below-normal priority."""
    if WIN:
        kw["creationflags"] = kw.get("creationflags", 0) | CREATE_NO_WINDOW | (
            BELOW_NORMAL_PRIORITY_CLASS if background else 0)
        p = subprocess.Popen(cmd, **kw)
    elif LINUX:
        if isinstance(cmd, (list, tuple)) and not kw.get("shell"):
            cmd = _linux_prefix(background) + list(cmd)
        p = _get_launcher().popen(cmd, **kw)   # (string / shell commands: kill_all at exit cleans up)
    else:
        if background:
            kw["preexec_fn"] = _preexec(10)
        p = subprocess.Popen(cmd, **kw)
    _live.add(p)
    return p


def run(cmd, timeout=None, background=False, text=False, input=None):
    """subprocess.run equivalent through popen(): output captured, killed on timeout."""
    p = popen(cmd, background=background, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
    try:
        out, err = p.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        raise
    finally:
        if p.poll() is None:
            p.kill()
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def kill_all():
    """Stop every child still running (called on exit; the job object / PDEATHSIG cover crashes)."""
    for p in list(_live):
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


atexit.register(kill_all)


def install_job_object():
    """Windows: put this process (and so every future descendant) in a kill-on-close job. True if active."""
    global _job
    if not WIN or _job is not None:
        return _job is not None
    try:
        import ctypes
        from ctypes import wintypes as W

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes",
                                                         "WriteBytes", "OtherBytes")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = W.HANDLE
        k32.GetCurrentProcess.restype = W.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = EXTENDED()
        info.Basic.LimitFlags = 0x2000                     # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(W.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(W.HANDLE(job))
            return False
        if not k32.AssignProcessToJobObject(W.HANDLE(job), W.HANDLE(k32.GetCurrentProcess())):
            k32.CloseHandle(W.HANDLE(job))                 # nesting refused: explicit cleanup still applies
            return False
        _job = job      # never closed: the handle closes when this process ends -> all descendants killed
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ worker processes (multiprocessing)
def worker_init(threads=1, background=True):
    """Initializer for batch / Manager processes: die with the parent, low priority, bounded threads."""
    if LINUX and _prctl is not None:
        _prctl(PR_SET_PDEATHSIG, SIGKILL)
    if background:
        try:
            if WIN:
                import ctypes
                k32 = ctypes.WinDLL("kernel32")
                k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
            else:
                os.nice(10)
        except Exception:
            pass
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["FX_DECODE_THREADS"] = str(threads)
    try:
        import cv2
        cv2.setNumThreads(threads)        # otherwise every worker's OpenCV uses ALL cores
    except Exception:
        pass


def terminate_pool(executor, grace=3.0):
    """Stop a ProcessPoolExecutor now: cancel queued work, give running workers `grace` seconds, then
    terminate them (Python's exit handler would otherwise wait for every running job to finish)."""
    if executor is None:
        return
    procs = list((getattr(executor, "_processes", None) or {}).values())   # before shutdown() drops it
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    deadline = time.monotonic() + grace
    for p in procs:
        try:
            p.join(max(0.0, deadline - time.monotonic()))
        except Exception:
            pass
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(2.0)
            if p.is_alive():
                p.kill()
        except Exception:
            pass


def start_manager(ctx, threads=1):
    """A multiprocessing Manager whose server process also dies with the app."""
    from multiprocessing.managers import SyncManager
    m = SyncManager(ctx=ctx)
    m.start(initializer=worker_init, initargs=(threads, True))
    return m


# ------------------------------------------------------------------ dynamic worker counts
def cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 2


def available_memory():
    """Bytes of RAM available now (best effort, no psutil needed)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        if LINUX:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        if WIN:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                      "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                      "ullAvailExtendedVirtual")]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return int(ms.ullAvailPhys)
    except Exception:
        pass
    return 4 * 2 ** 30


def cpu_busy_fraction():
    """Share of the CPU other programs use right now (0-1), or None if unknown."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.2) / 100.0
    except Exception:
        pass
    try:
        return min(1.0, os.getloadavg()[0] / cpu_count())
    except Exception:
        return None


SHARE = {"light": 0.25, "balanced": 0.5, "max": 1.0}


def cpu_budget(share="balanced"):
    """Cores FrameExtractor may use without making the PC sluggish: a share of the cores, minus what other
    programs are busy with right now."""
    cores = cpu_count()
    allowed = cores * SHARE.get(share, 0.5)
    busy = cpu_busy_fraction()
    if busy is not None:
        allowed = min(allowed, max(1.0, cores * (1.0 - busy) - 0.5))
    return max(1, int(allowed))


def decode_threads_for(width, height):
    px = width * height
    return 1 if px <= 1280 * 720 else 2 if px <= 1920 * 1080 else 4 if px <= 3840 * 2160 else 6


def plan_batch_workers(videos, share="balanced", ram_reserve=2 * 2 ** 30):
    """videos: [(width, height)] of the queued jobs -> (workers, threads_per_worker) within CPU and RAM budgets."""
    if not videos:
        return 1, 1
    w, h = max(videos, key=lambda v: v[0] * v[1])
    threads = decode_threads_for(w, h)
    budget = cpu_budget(share)
    by_cpu = max(1, budget // threads)
    per_worker_ram = w * h * 3 * 40 + 300 * 2 ** 20
    by_ram = max(1, int((available_memory() - ram_reserve) // per_worker_ram))
    workers = max(1, min(by_cpu, by_ram, len(videos) * 8))
    threads = max(1, min(threads, budget // workers)) if workers > 1 else min(threads, budget)
    return workers, threads


def plan_analysis_workers(width, height, samples, share="balanced"):
    """Timeline analysis: enough parallel decoders for ~1-2 s, within the CPU budget."""
    budget = cpu_budget(share)
    per = decode_threads_for(width, height) / 2.0
    want = max(1, min(4, int(samples / 60) + 1))
    return max(1, min(want, int(budget / max(1.0, per))))


def decode_ahead_budget():
    """RAM parallel playback decoders may hold ahead of the playhead: a third of what is free, at most 6 GB.
    (Each decoder necessarily buffers about one GOP - 8K frames are ~50 MB each.)"""
    return int(min(6 * 2 ** 30, available_memory() // 3))


def plan_parallel_decoders(need_fps, single_fps, frame_bytes, segment_frames, share="max", ram_budget=None):
    """Playback: just enough parallel decoders to reach the video's rate (+20 %) - no more - within the CPU
    and RAM budgets. Playback is foreground work, so it may use the whole machine ('max') by default."""
    if not single_fps or single_fps >= need_fps * 1.1:
        return 1
    want = math.ceil(need_fps * 1.2 / single_fps)
    budget = ram_budget or decode_ahead_budget()
    by_ram = max(1, int(budget // max(1, frame_bytes * segment_frames)))
    return max(1, min(want, cpu_budget(share), by_ram, 8))
