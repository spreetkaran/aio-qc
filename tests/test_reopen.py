# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Opening the dataset that is already open (Recent, Open, a start page) must not wait for its own cache, run
uncached, or leave threads behind."""
import time

from PyQt6.QtCore import QThread

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _wait_index(w, qtbot):
    qtbot.waitUntil(lambda: w.session is not None and w.session.index_complete, timeout=30000)


def test_reopen_same_dataset_keeps_its_cache_and_is_quick(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    _wait_index(w, qtbot)
    for how in ("recent", "open", "while indexing"):
        t = time.time()
        if how == "recent":
            w._build_recent_menu()
            next(a for a in w.recent_menu.actions() if a.toolTip() == str(root)).trigger()
        elif how == "open":
            w.open_dataset(root)
        else:
            w.open_dataset(root)
            w.open_dataset(root)                                 # the first one is still indexing
        assert time.time() - t < 10, how                          # it waited 30 s for the old session's lock
        assert w.session.cache is not None, how                   # ... and then ran without its cache
        _wait_index(w, qtbot)


def test_cache_flush_releases_the_write_lock(tmp_path):
    from AIO_QC.core.cache import DatasetCache
    root = FX.labelme(tmp_path / "ds")
    c = DatasetCache(root)
    c.db.execute("UPDATE meta SET v=v WHERE k='schema'")          # like the "last used" marks of a cache hit
    assert c.db.in_transaction
    c.flush()
    assert not c.db.in_transaction
    other = DatasetCache(root)                                    # opens at once: nothing is holding the lock
    assert other.load_all_labels() == {}
    other.close()
    c.close()


def test_folder_scan_stops_when_asked(tmp_path):
    from AIO_QC.core.watch import snapshot
    for i in range(10):
        (tmp_path / f"d{i}").mkdir()
        (tmp_path / f"d{i}" / "a.json").write_text("{}")
    assert len(snapshot(tmp_path, {".json"})) == 10
    assert snapshot(tmp_path, {".json"}, should_stop=lambda: True) is None


def test_reopening_leaves_no_threads_behind(shell, qtbot, tmp_path):  # noqa: F811
    root = FX.labelme(tmp_path / "ds")
    w = _open(shell, qtbot, root)
    _wait_index(w, qtbot)
    qtbot.wait(2500)                                              # the first open's housekeeping has run

    def threads():
        return len([t for t in w.findChildren(QThread)])
    w.open_dataset(root)
    _wait_index(w, qtbot)
    qtbot.wait(2500)
    base = threads()
    for _ in range(3):
        w.open_dataset(root)
        _wait_index(w, qtbot)
        qtbot.wait(2500)
    qtbot.waitUntil(lambda: threads() <= base, timeout=10000)     # finished workers are freed, not kept per open
