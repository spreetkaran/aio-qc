# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Background work steps aside while the user moves the view.

Python threads share one interpreter lock: a background loop (the image integrity sweep, label indexing) running
while the canvas glides takes turns with the UI thread and costs it whole frames — zooming stuttered for the first
minute or two after opening a large dataset, until that work was done. The canvas marks the moments it animates
(`interacting`); background loops call `yield_to_ui()` between items and simply wait those moments out."""
import time

_until = 0.0


def interacting(seconds: float = 0.3):
    """The UI is animating for (at least) the next `seconds`."""
    global _until
    t = time.monotonic() + max(0.0, seconds)
    if t > _until:
        _until = t


def busy() -> bool:
    return time.monotonic() < _until


def yield_to_ui(max_wait: float = 3.0, stop=None):
    """Wait while the UI animates (at most `max_wait` s at a time, so work never stalls under endless motion)."""
    if time.monotonic() >= _until:
        return
    end = time.monotonic() + max_wait
    while time.monotonic() < min(_until, end):
        if stop is not None and stop():
            return
        time.sleep(0.015)
