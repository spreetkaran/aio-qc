# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Streaming detection analysis (core.testing.stream): look-ahead, adaptive frame step, interpolation, backfill,
priority for a paused frame, the shared prediction cache, failure isolation and frame-exact sync with the player's
decoder."""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pytest

from .fx_fixtures import D

VIDEO = os.path.join(D, "a.mp4")            # 200 frames, 25 fps


class FakeDet:
    names = {0: "thing", 1: "other"}
    provider = "fake"

    def __init__(self, delay=0.0, fail=None, oom_above=None, seen=None):
        self.delay, self.fail, self.oom_above, self.seen = delay, fail, oom_above, seen

    def detect_batch(self, frames, conf=None):
        if self.fail:
            raise RuntimeError(self.fail)
        if self.oom_above and len(frames) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        time.sleep(self.delay * len(frames))
        out = []
        for f in frames:
            if self.seen is not None:
                self.seen.append(hashlib.md5(f.tobytes()).hexdigest())
            m = float(f.mean())                         # a box that depends on the frame: proves frame identity
            out.append(np.array([[10, 10, 50 + m, 60, 0.9, 0]], np.float32))
        return out


class FakeSpec:
    def __init__(self, name, det=None, load_error=None, conf=0.25):
        self.path, self.label, self.conf = f"/fake/{name}.onnx", name, conf
        self.key = hashlib.sha1(name.encode()).hexdigest()[:16]
        self.det, self.load_error, self.loads = det or FakeDet(), load_error, 0

    def load(self):
        self.loads += 1
        if self.load_error:
            raise FileNotFoundError(self.load_error)
        return self.det

    def to_dict(self):
        return {"path": self.path, "label": self.label}


def _wait(cond, t=30.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_interpolation_pairs_by_class_and_iou():
    from AIO_QC.core.testing.stream import interpolate
    a = np.array([[0, 0, 10, 10, 0.8, 1], [100, 100, 120, 120, 0.9, 2]], np.float32)
    b = np.array([[10, 0, 20, 10, 0.6, 1], [300, 300, 320, 320, 0.7, 3]], np.float32)
    mid = interpolate(a, b, 0.5)
    moved = mid[mid[:, 5] == 1]
    assert len(moved) == 1 and np.allclose(moved[0, :5], [5, 0, 15, 10, 0.7])
    assert 2 in mid[:, 5] and 3 not in mid[:, 5]          # unpaired: the earlier one in the first half
    late = interpolate(a, b, 0.75)
    assert 3 in late[:, 5] and 2 not in late[:, 5]
    assert interpolate(np.zeros((0, 6)), np.zeros((0, 6)), 0.3).shape == (0, 6)


def test_adaptive_stride():
    from AIO_QC.core.testing.stream import adaptive_stride
    assert adaptive_stride(25, 200, 1) == 1                # fast GPU: every frame
    assert adaptive_stride(30, 20, 1) == 2                 # 20 / s for 30 fps: every 2nd
    assert adaptive_stride(60, 10, 1) == 7
    assert adaptive_stride(60, 1, 1) == 8                  # capped
    assert adaptive_stride(25, 100, 3) == 2                # finer one step at a time
    assert adaptive_stride(25, 28, 2) == 2                 # no headroom: stays


def test_best_mode_analyses_every_frame_and_caches(tmp_path):
    from AIO_QC.core.testing.runner import ModelSpec
    from AIO_QC.core.testing.stream import StreamAnalyser
    specs = [ModelSpec(os.path.join(D, "models", m), imgsz=320) for m in ("dummy_a.onnx", "dummy_b.onnx")]
    cache = str(tmp_path / "pred.db")
    a = StreamAnalyser(specs, VIDEO, cache, mode="best").start()
    assert _wait(lambda: a.coverage() >= 1.0, 120), a.status()
    a.stop()
    assert a.status()["errors"] == {} and a.first == 0 and a.last == 199
    per, how = a.detections(77)
    assert how == "exact" and set(per) == {s.key for s in specs}
    # a second analyser of the same video finds everything in the cache and never loads a model
    for s in specs:
        s.load = lambda: (_ for _ in ()).throw(AssertionError("model loaded although cached"))
    b = StreamAnalyser(specs, VIDEO, cache, mode="best").start()
    assert _wait(lambda: b.done.is_set() or b.phase == "done", 20)
    assert b.coverage() >= 1.0 and b.detections(150)[1] == "exact"
    b.stop()


def test_balanced_keeps_up_with_a_slow_model_then_backfills(tmp_path):
    from AIO_QC.core.testing.stream import StreamAnalyser
    spec = FakeSpec("slow", FakeDet(delay=0.06))          # ~16 frames / s for a 25 fps video
    a = StreamAnalyser([spec], VIDEO, str(tmp_path / "p.db"), mode="balanced", lead_s=1.0, batch=4)
    a.start()
    assert a.ready.wait(10)
    a.set_playhead(0, playing=True)
    assert _wait(lambda: a.ready_until(0) >= 12, 15)       # the short start-up buffer
    t0, misses, shown = time.time(), 0, 0
    while time.time() - t0 < 3.0:                          # play in real time for 3 s
        idx = int((time.time() - t0) * a.fps)
        a.set_playhead(idx, playing=True)
        per, how = a.detections(idx)
        shown += 1
        misses += how is None
        time.sleep(0.04)
    st = a.status()
    assert st["stride"] >= 2, st                           # coarser than every frame: it adapted
    assert misses <= shown * 0.1, (misses, shown, st)      # boxes on screen practically all the time
    a.set_playhead(idx, playing=False)                     # paused: backfill fills every skipped frame
    assert _wait(lambda: a.coverage() >= 1.0, 60), a.status()
    assert all(a.detections(i)[1] == "exact" for i in range(0, 200, 7))
    a.stop()


def test_paused_frame_jumps_the_queue(tmp_path):
    from AIO_QC.core.testing.stream import StreamAnalyser
    spec = FakeSpec("slowish", FakeDet(delay=0.03))
    a = StreamAnalyser([spec], VIDEO, str(tmp_path / "p.db"), mode="best", batch=4).start()
    assert a.ready.wait(10)
    a.set_playhead(0, playing=False)
    time.sleep(0.2)
    assert not a.has(180)
    a.request_exact(180)
    assert _wait(lambda: a.has(180), 3), a.status()
    a.set_playhead(120, playing=True)                      # a seek moves the look-ahead
    assert _wait(lambda: a.has(125), 3)
    a.stop()


def test_failing_model_is_isolated_and_oom_retries_smaller(tmp_path):
    from AIO_QC.core.testing.stream import StreamAnalyser
    good = FakeSpec("good", FakeDet(oom_above=2))
    missing = FakeSpec("missing", load_error="no such file")
    broken = FakeSpec("broken", FakeDet(fail="bad output shape"))
    logs = []
    a = StreamAnalyser([good, missing, broken], VIDEO, str(tmp_path / "p.db"), mode="best", batch=8,
                       log=logs.append).start()
    assert _wait(lambda: a.coverage() >= 1.0, 60), a.status()
    a.stop()
    errs = a.status()["errors"]
    assert set(errs) == {missing.key, broken.key}
    assert "no such file" in errs[missing.key] and "bad output shape" in errs[broken.key]
    assert any("out of memory" in m for m in logs)
    assert a.detections(10)[1] == "exact" and set(a.detections(10)[0]) == {good.key}


def test_stop_is_quick_and_resumes_from_the_cache(tmp_path):
    from AIO_QC.core.testing.stream import StreamAnalyser
    cache = str(tmp_path / "p.db")
    spec = FakeSpec("resume", FakeDet(delay=0.02))
    a = StreamAnalyser([spec], VIDEO, cache, mode="best", batch=4).start()
    assert _wait(lambda: a.coverage() > 0.2, 20)
    t = time.time()
    a.stop()
    assert time.time() - t < 2.0 and not a.running
    done_before = a.coverage()
    b = StreamAnalyser([spec], VIDEO, cache, mode="best", batch=4).start()
    assert b.ready.wait(10) and b.coverage() >= done_before - 0.03
    assert _wait(lambda: b.coverage() >= 1.0, 30)
    b.stop()


@pytest.mark.parametrize("backend", ["opencv", "auto"])
def test_frame_numbers_match_the_players_decoder(tmp_path, backend):
    """The analyser decodes with the player's backend: frame i it analysed is frame i the player shows - also after
    seeks and stride jumps."""
    from AIO_QC.core.testing.stream import StreamAnalyser
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    seen = []
    spec = FakeSpec(f"sync-{backend}", FakeDet(seen=seen))
    a = StreamAnalyser([spec], VIDEO, str(tmp_path / "p.db"), mode="fixed", stride=3, batch=4,
                       backend=backend).start()
    assert a.ready.wait(10)
    a.set_playhead(0, playing=True)
    assert _wait(lambda: a.coverage() >= 1.0, 60)
    a.stop()
    src = open_source(VIDEO, backend, hw_accel=False)
    try:
        for i in (0, 3, 4, 50, 101, 150, 199):
            src.seek(i)
            ok, img = src.read()
            want = np.array([[10, 10, 50 + float(img.mean()), 60, 0.9, 0]], np.float32)
            assert np.allclose(a._store[spec.key][i], want, atol=1e-3), i
    finally:
        src.release()
