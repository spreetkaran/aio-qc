# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A running background thread is never destroyed (Qt would abort the app): not with its window, not when Python drops
it, not by deleteLater; the dataset databases avoid memory-mapped journals off local disks."""
import gc
import sqlite3
import threading
import time

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from AIO_QC.widgets import lifecycle
from AIO_QC.widgets.lifecycle import RetainedThread


class Slow(RetainedThread, QThread):
    def __init__(self, gate, parent=None):
        super().__init__(parent)
        self.gate = gate
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        t = time.time()
        while not self.gate.is_set() and time.time() - t < 10:
            time.sleep(0.01)


def _spin(ms=100):
    t = time.time()
    while time.time() - t < ms / 1000:
        QApplication.processEvents()
        time.sleep(0.005)


def test_window_deleted_while_its_thread_runs(qtbot):
    gate = threading.Event()
    dlg = QDialog()
    inner = QWidget(dlg)
    t = Slow(gate, inner)
    t.start()
    assert t.parent() is None and lifecycle.owner_of(t) is inner and t.objectName() == "Slow"
    dlg.deleteLater()                                 # used to delete the running thread with it: abort
    _spin(200)
    assert t.isRunning()
    gate.set()
    assert t.wait(5000)


def test_stop_threads_still_finds_detached_threads(qtbot):
    gate = threading.Event()
    dlg = QDialog()
    t = Slow(gate, QWidget(dlg))
    t.start()
    gate.set()
    assert lifecycle.stop_threads(dlg, "test") == 1 and t.stopped and not t.isRunning()
    dlg.deleteLater()
    _spin(50)


def test_dropped_reference_and_delete_later_wait_for_the_end(qtbot):
    gate = threading.Event()
    t = Slow(gate)
    t.start()
    t.deleteLater()                                   # deferred while it runs
    ref = [t]
    del t
    ref.clear()
    gc.collect()
    _spin(150)
    assert any(isinstance(x, Slow) and x.isRunning() for x in lifecycle.still_running())
    gate.set()
    _spin(400)
    assert not [x for x in lifecycle.still_running() if isinstance(x, Slow)]
    lifecycle.final_exit(0)                           # nothing running: returns


def test_journal_off_local_disks(tmp_path, monkeypatch):
    from AIO_QC.core import sqlite_util
    from AIO_QC.core.cache import DatasetCache as Cache
    monkeypatch.setenv("AIOQC_SQLITE_STORAGE", "network")
    assert sqlite_util.journal_mode(tmp_path) == "DELETE"
    db = sqlite3.connect(str(tmp_path / "x.sqlite"))
    db.execute("PRAGMA journal_mode=WAL")             # an existing WAL database is switched back
    assert sqlite_util.configure(db, tmp_path / "x.sqlite") == "DELETE"
    db.close()
    (tmp_path / "ds").mkdir()
    c = Cache(tmp_path / "ds")
    mode = c.db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "delete"
    c.close()
    assert not list((tmp_path / "ds").rglob("*-shm"))
    monkeypatch.setenv("AIOQC_SQLITE_STORAGE", "local")
    assert sqlite_util.journal_mode(tmp_path) == "WAL"
    monkeypatch.delenv("AIOQC_SQLITE_STORAGE")
    assert sqlite_util.storage_kind(tmp_path) in ("local", "network", "removable")
