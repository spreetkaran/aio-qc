# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Running out of memory while drawing an image is never fatal: the fast view stays, caches are freed, one notice."""
import cv2
import pytest
from PyQt6.QtGui import QImage

from AIO_QC.utils import memory as M
from AIO_QC.widgets import canvas as C

from . import fixtures as FX
from .test_aio_image import _open, shell  # noqa: F401


def _oom(*_a, **_k):
    raise cv2.error("OpenCV(4.12.0) alloc.cpp:73: error: (-4:Insufficient memory) Failed to allocate 4925440 bytes "
                    "in function 'cv::OutOfMemoryError'")


def test_null_qimage_is_a_memory_error():
    with pytest.raises(MemoryError):
        C._qimage_array(QImage(), write=True)
    assert C._is_oom(MemoryError()) and C._is_oom(cv2.error("(-4:Insufficient memory) Failed to allocate 1 bytes"))
    assert not C._is_oom(ValueError("nope"))


@pytest.mark.parametrize("where", ["resize", "warp", "qimage"])
def test_hq_render_out_of_memory(shell, qtbot, tmp_path, monkeypatch, where):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 6))
    w.resize(1200, 800)
    qtbot.waitUntil(lambda: w.current_key is not None)
    item = w.canvas.pixmap_item
    w.canvas.animate_to(0.3 if where == "resize" else 6.0, duration=0)
    if where == "resize":
        monkeypatch.setattr(cv2, "resize", _oom)
    elif where == "warp":
        monkeypatch.setattr(cv2, "warpAffine", _oom)
    else:
        orig = C._qimage_array
        monkeypatch.setattr(C, "_qimage_array", lambda img, write=False:
                            orig(QImage(), True) if write else orig(img, write))
    seen = []
    w.canvas.memoryLow.connect(seen.append)
    item.refine()                                           # was: an unhandled exception
    assert seen == ["high-quality view"] and item.hq is None and item._backoff > 0
    assert w.pixmap_cache.total >= 0 and w.toast.find("low_memory") is not None
    item.refine()                                           # backing off: no retry storm, no second notice
    assert seen == ["high-quality view"]


def test_brightness_out_of_memory_shows_the_image(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 4))
    qtbot.waitUntil(lambda: w.current_key is not None)
    c = w.canvas
    orig = C._qimage_array
    monkeypatch.setattr(C, "_qimage_array", lambda img, write=False:
                        orig(QImage(), True) if write else orig(img, write))
    seen = []
    c.memoryLow.connect(seen.append)
    c.display["brightness"] = 40
    c._render_display()
    assert seen == ["brightness / contrast"]
    assert c.pixmap_item.pixmap().cacheKey() == c._original.cacheKey()      # shown unadjusted, not blank


def test_cache_budget_fits_the_computer(monkeypatch):
    class VM:
        total, available = 4 * 2 ** 30, 300 * 2 ** 20
    monkeypatch.setattr(M, "_vm", lambda: VM)
    assert M.cache_budget(768 * 2 ** 20) == int(4 * 2 ** 30 * 0.12)
    assert M.is_low()
    monkeypatch.setattr(M, "_vm", lambda: None)
    assert M.cache_budget(768 * 2 ** 20) == 768 * 2 ** 20 and not M.is_low()


def test_systemerror_caused_by_numpy_memory_error():
    """OpenCV's binding wraps numpy's _ArrayMemoryError: "returned a result with an exception set"."""
    try:
        try:
            raise MemoryError("Unable to allocate 4.70 MiB for an array with shape (832, 1480, 4)")
        except MemoryError as inner:
            raise SystemError("<built-in function resize> returned a result with an exception set") from inner
    except SystemError as e:
        assert M.is_oom(e) and C._is_oom(e)
    assert not M.is_oom(SystemError("something else"))


def test_wrapped_error_in_hq_render(shell, qtbot, tmp_path, monkeypatch):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 4))
    qtbot.waitUntil(lambda: w.current_key is not None)
    w.canvas.animate_to(0.3, duration=0)

    def wrapped(*_a, **_k):
        try:
            raise MemoryError("Unable to allocate 4.70 MiB")
        except MemoryError as inner:
            raise SystemError("<built-in function resize> returned a result with an exception set") from inner
    monkeypatch.setattr(cv2, "resize", wrapped)
    seen = []
    w.canvas.memoryLow.connect(seen.append)
    w.canvas.pixmap_item.refine()
    assert seen == ["high-quality view"]


def test_shared_cache_budget():
    from PyQt6.QtGui import QImage
    from AIO_QC.utils import io as U
    import weakref
    old, old_caches = U.BUDGET._limit, U.BUDGET.caches
    U.BUDGET._limit = 3 * 1000 * 1000 * 4                    # room for three 1000 × 1000 images
    U.BUDGET.caches = weakref.WeakSet()                       # only this test's caches
    try:
        a, b = U.LRUCache(capacity=50), U.LRUCache(capacity=50)
        img = QImage(1000, 1000, QImage.Format.Format_RGB32)
        assert U.cost_of(img) == 4 * 10 ** 6 and U.cost_of((img, {"x": 1})) == 4 * 10 ** 6
        for i in range(3):
            a.put(i, QImage(img))
        b.put("x", QImage(img))                               # over the shared budget: a's oldest goes
        assert U.BUDGET.total() <= U.BUDGET._limit and 0 not in a and "x" in b
        U.BUDGET.trim(0.5)
        assert len(a) >= 1 and len(b) == 1
    finally:
        U.BUDGET._limit, U.BUDGET.caches = old, old_caches


def test_closed_tools_and_modal_dialogs_are_freed(shell, qtbot, tmp_path):  # noqa: F811
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QDialog
    from AIO_QC.widgets import ui_style
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 4))
    for _ in range(3):
        w.open_tool("stats")
        w.open_dialogs["stats"].close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qtbot.wait(50)
    assert not [d for d in w.findChildren(QDialog) if type(d).__name__ == "StatsDialog"]
    # what QDialog.exec now does after the dialog closes (the test harness stubs exec itself)
    assert ui_style._ORIG_EXEC is not None
    d, k = QDialog(w), QDialog(w)
    k.keep_after_exec = True
    for x in (d, k):
        x.open()
        x.accept()
    assert ui_style.release(d) and not ui_style.release(k)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qtbot.wait(20)
    alive = w.findChildren(QDialog)
    assert k in alive and all(x is not d for x in alive)


def test_index_counters_shared(shell, qtbot, tmp_path):  # noqa: F811
    w = _open(shell, qtbot, FX.labelme(tmp_path / "ds", 12))
    s = w.session
    vals = list(s.raw_index.values())
    assert len({id(v) for v in vals}) < len(vals)              # identical label counts share one object
    assert all(s.index[k] is s.raw_index[k] for k in s.raw_index)   # no mapping: no second copy
