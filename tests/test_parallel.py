# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""core.parallel: in order, exceptions per item, threads only for slow files; Placer counts and collects errors."""
import threading
import time

from AIO_QC.core import parallel as PL


def test_ordered_map_fast_items_stay_in_one_thread():
    seen = set()

    def fn(x):
        seen.add(threading.get_ident())
        if x == 5:
            raise ValueError("five")
        return x * 2
    out = list(PL.ordered_map(fn, list(range(500)), workers=4))
    assert [x for x, _r in out] == list(range(500))
    assert isinstance(out[5][1], ValueError) and out[6][1] == 12
    assert seen == {threading.get_ident()}


def test_ordered_map_slow_items_use_threads_and_keep_order():
    seen = set()

    def fn(x):
        seen.add(threading.get_ident())
        time.sleep(0.003)                         # a file waiting on storage
        return -x
    t = time.perf_counter()
    out = list(PL.ordered_map(fn, list(range(600)), workers=6, probe=20))
    dt = time.perf_counter() - t
    assert [r for _x, r in out] == [-x for x in range(600)]
    assert len(seen) > 1 and dt < 600 * 0.003 * 0.7          # faster than one at a time


def test_placer_counts_errors_and_switches_to_threads(tmp_path):
    def ok(i):
        time.sleep(0.003)
        return "copy" if i % 2 else "link"

    def bad(_i):
        raise OSError("disk full")
    P = PL.Placer(workers=4, probe=8)
    for i in range(100):
        P.put(ok, i, name=f"f{i}")
    assert P.threaded
    P.put(bad, 1, name="broken.jpg")
    P.close()
    assert P.counts == {"link": 50, "copy": 50} and P.errors == ["broken.jpg: could not be written (disk full)"]
