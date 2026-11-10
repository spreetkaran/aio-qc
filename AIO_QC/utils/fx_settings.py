# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Settings of the Frame Extractor and of Testing inside AIO QC's one settings store, and the one-time migration from
the standalone FrameExtractor's QSettings("FrameExtractor", "FrameExtractor").

    fx_settings()        AIO QC settings, group "fx"       (everything the Frame Extractor stores)
    testing_settings()   AIO QC settings, group "testing"  (the former Model Lab: "lab/<key>" -> "testing/<key>")
    migrate_settings()   copies the old keys once:  lab/*        -> testing/*
                                                    ui/animations -> ui/animations (only if AIO QC has none yet)
                                                    ui/theme      -> dropped (one theme: AIO QC's)
                                                    anything else -> fx/*
                         existing AIO QC values always win; the old store is left untouched.
"""
from __future__ import annotations

from loguru import logger
from PyQt6.QtCore import QSettings

from AIO_QC.utils.qt_utils import app_settings

DONE_KEY = "migrations/frame_extractor_settings"
_checked = [False]


def _group(name: str) -> QSettings:
    migrate_settings()
    s = app_settings()
    s.beginGroup(name)
    return s


def fx_settings() -> QSettings:
    return _group("fx")


def testing_settings() -> QSettings:
    return _group("testing")


TRANSIENT = {"ui/theme", "crash/sentinel", "ui/gpu_failed", "safe/last_crash"}   # state of the old app's last runs


def map_key(k: str) -> str | None:
    if k in TRANSIENT:
        return None
    if k == "ui/animations":
        return "ui/animations"
    if k.startswith("lab/"):
        return "testing/" + k[4:]
    return "fx/" + k


GPU_RETRY_KEY = "migrations/fx_gpu_retry_2_1"


def _gpu_retry_once(new: QSettings):
    """2.1.0 could turn the Frame Extractor's GPU renderer off on Windows (a fade made its window layered, the
    canvas failed, and 'GPU failed' was remembered) or carry the old app's crash state over: forget both once."""
    if new.value(GPU_RETRY_KEY):
        return
    for k in ("fx/ui/gpu_failed", "fx/crash/sentinel", "fx/safe/last_crash"):
        new.remove(k)
    new.setValue(GPU_RETRY_KEY, 1)


def migrate_settings(old: QSettings | None = None) -> int:
    """Copy the standalone app's settings once. Returns how many keys were copied."""
    if _checked[0]:
        return 0
    _checked[0] = True
    new = app_settings()
    _gpu_retry_once(new)
    if new.value(DONE_KEY):
        return 0
    old = old or QSettings("FrameExtractor", "FrameExtractor")
    n = 0
    for k in old.allKeys():
        t = map_key(k)
        if t is None or new.contains(t):
            continue
        new.setValue(t, old.value(k))
        n += 1
    new.setValue(DONE_KEY, 1)
    new.sync()
    if n:
        logger.info(f"[frame extractor] {n} setting(s) taken over from the standalone FrameExtractor")
    return n
