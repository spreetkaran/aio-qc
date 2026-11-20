# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Cycles are collected on the GUI thread only: a widget caught in a cycle never dies on a worker thread."""
import gc
import threading
import weakref

from PyQt6.QtWidgets import QWidget


def test_cycles_die_on_the_gui_thread(qapp, qtbot):
    from AIO_QC.utils import gui_gc
    col = gui_gc.install(qapp)
    assert not gc.isenabled() and gui_gc.installed()
    died_on = []

    class Cyclic(QWidget):
        def __init__(self):
            super().__init__()                      # Python owns it (no parent)
            self.me = self                          # a cycle: only the collector frees it

    refs = [weakref.ref(Cyclic(), lambda _r: died_on.append(threading.get_ident())) for _ in range(20)]

    def churn():                                    # a worker that allocates a lot (would trigger automatic gc)
        junk = []
        for i in range(200_000):
            junk.append([i])
            if len(junk) > 1000:
                junk.clear()
    t = threading.Thread(target=churn)
    t.start()
    t.join()
    assert not died_on                              # nothing was collected on the worker
    assert gui_gc.pause(True) and col.paused
    gui_gc.pause(False)
    gc.collect()                                    # the GUI thread collects (here: at once)
    assert len(died_on) == 20 and set(died_on) == {threading.get_ident()} and not any(r() for r in refs)
