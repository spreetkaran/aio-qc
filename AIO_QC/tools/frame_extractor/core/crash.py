# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Crash handling of the Frame Extractor inside AIO QC.

AIO QC's main() owns everything process-wide - exactly once (widgets/lifecycle.py + __main__.py):
  * sys.excepthook / threading.excepthook / sys.unraisablehook (exceptions in Qt slots are logged, the app keeps
    running, the user sees them in AIO QC's error log),
  * faulthandler (native crashes -> aio_qc_crash.log in the log folder),
  * Qt's message handler, and the loguru sinks (one log file: aio_qc_tool.log).

The functions below keep the Frame Extractor's former entry points working as no-ops, and tell it where AIO QC's
log and crash files are (the Frame Extractor's own crash sentinel - which turns on safe settings after a crash while
opening or playing a video - is unchanged and lives in its settings).
"""
import glob
import os

from AIO_QC.core.appdirs import log_dir as _aio_log_dir

LOG_DIR = str(_aio_log_dir())
LOG_FILE = os.path.join(LOG_DIR, "aio_qc_tool.log")


def setup_logging(level="DEBUG"):              # AIO QC's main() configured loguru
    return LOG_DIR


def enable_native_crash_log():                 # faulthandler: AIO QC's lifecycle.install()
    return os.path.join(LOG_DIR, "aio_qc_crash.log")


def cleanup_empty_crash_logs():                # lifecycle removes an empty crash log at exit
    return None


def install_exception_hooks():                 # AIO QC's excepthooks (one per process)
    return None


def install_qt_message_handler():              # AIO QC's Qt message handler
    return None


def latest_crash_log():
    """The newest native crash log of a previous run (lifecycle renames it to aio_qc_crash_<time>.log)."""
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "aio_qc_crash_*.log")), key=os.path.getmtime)
    return logs[-1] if logs else None
