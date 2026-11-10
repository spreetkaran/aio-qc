# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Fast timeline sampling (Qt-free).

Cost must not grow with video length (target: 1-2 s typical, <= 5 s for 4-5 hour files), so:
  * never read the whole file - no packet scans; the container index is used for seeking
  * sample keyframes: one decode per sample, no decoding forward from a keyframe
      PyAV   : seek(backward to keyframe) + decode that keyframe only (non-key frames skipped)
      ffmpeg : one short process per sample: input seek + keyframe-only decode + scale in C
      OpenCV : last resort (its seek always decodes from the previous keyframe)
  * work at small size: thumbnail + sharpness from a scaled copy (PyAV: swscale straight to 640 px)
  * several decoders in parallel threads (decoders release the GIL)
  * coarse-to-fine order, so the strip covers the whole video within a fraction of a second
"""
import math
import shutil
import threading
import time
from collections import deque

import cv2
import numpy as np

from AIO_QC.tools.frame_extractor.core.metrics import sharpness
from AIO_QC.tools.frame_extractor.core.sources import HAVE_PYAV, _ROTATE
from AIO_QC.core import procs



THUMB_W = 192

if HAVE_PYAV:
    import av


def plan_grid(frame_count, duration, per_sec=2.0, lo=60, hi=320):
    n = int(min(frame_count, max(lo, min(hi, duration * per_sec))))
    if n <= 0:
        return []
    return sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, n)))


def coarse_to_fine(n):
    seen, order = set(), []
    stride = 1 << max(0, int(math.log2(max(1, n))) - 1)
    while stride >= 1:
        for i in range(0, n, stride):
            if i not in seen:
                seen.add(i)
                order.append(i)
        stride //= 2
    return order


def estimate_gop(path, ffprobe=None, packets=600, timeout=10):
    """Keyframe interval (frames) from the first packets only - milliseconds, not a file scan.
    Returns (gop or None, regular: bool)."""
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        return None, False
    try:
        out = procs.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{packets}",
                              "-show_entries", "packet=flags", "-of", "csv=p=0", path],
                             text=True, timeout=timeout).stdout
    except Exception:
        return None, False
    keys = [i for i, l in enumerate(out.splitlines()) if l.strip().upper().startswith("K")]
    if len(keys) < 2:
        return None, False
    gaps = np.diff(keys)
    g = int(np.median(gaps))
    regular = bool(np.all(np.abs(gaps - g) <= 1))
    return g, regular


class _Result:
    __slots__ = ("slot", "idx", "thumb", "sharp", "motion", "hist", "full")

    def __init__(self, slot, idx, thumb, sharp, motion, hist, full=None):
        self.slot, self.idx, self.thumb, self.sharp = slot, idx, thumb, sharp
        self.motion, self.hist, self.full = motion, hist, full


def _hist(bgr):
    small = cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA) if bgr.shape[1] > 96 else bgr
    h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h)


def _thumb(bgr):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (THUMB_W, max(2, int(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)


class OpenCVSampler:
    def __init__(self, path, gop=None, keep_full=False):
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if hasattr(cv2, "CAP_PROP_N_THREADS"):
            self.cap.set(cv2.CAP_PROP_N_THREADS, 2)     # parallelism comes from several samplers
        self.gop = gop
        self.keep_full = keep_full
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def snap(self, idx):
        if not self.gop:
            return idx
        k = int(round(idx / self.gop)) * self.gop
        return min(k, max(0, self.n - 1)) if self.n else k

    def sample(self, slot):
        idx = self.snap(slot)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = self.cap.read()
        if not ok:
            return None
        return _Result(slot, idx, _thumb(img), sharpness(img), None, _hist(img), img if self.keep_full else None)

    def close(self):
        self.cap.release()


class FfmpegSampler:
    """One ffmpeg process per sample: seek to the keyframe before t, decode only that frame, scale to 640 px.
    Cost doesn't depend on the video's length. showinfo reports the keyframe's real timestamp."""
    _pts = __import__("re").compile(r"pts_time:\s*(-?[\d.]+)")

    def __init__(self, ffmpeg, path, width, height, time_of, index_of, keep_full=False, start_time=0.0):
        self.ffmpeg, self.path = ffmpeg, path
        self.time_of, self.index_of = time_of, index_of
        self.ow = min(1280 if keep_full else 640, width)
        self.oh = max(2, int(round(height * self.ow / width / 2)) * 2)
        self.start = start_time
        self.keep_full = keep_full

    def sample(self, slot):
        t = max(0.0, self.time_of(slot))
        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-threads", "2", "-skip_frame", "nokey",
               "-ss", f"{t:.3f}", "-noaccurate_seek", "-copyts", "-i", self.path, "-an", "-sn", "-frames:v", "1",
               "-vf", f"scale={self.ow}:{self.oh}:flags=area,showinfo", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        r = procs.run(cmd, timeout=30, background=True)
        n = self.ow * self.oh * 3
        if len(r.stdout) < n:
            return None
        img = np.frombuffer(r.stdout[:n], np.uint8).reshape(self.oh, self.ow, 3)
        m = self._pts.search(r.stderr.decode("utf-8", "replace"))
        # showinfo prints pts rounded to 5 decimals: allow 0.1 ms so the frame isn't mistaken for its predecessor
        idx = max(0, self.index_of(float(m.group(1)) - self.start + 1e-4)) if m else slot
        return _Result(slot, idx, _thumb(img), sharpness(img), None, _hist(img), img if self.keep_full else None)

    def close(self):
        pass


def probe_start_time(path, ffprobe=None):
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = procs.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
                              "-of", "csv=p=0", path], text=True, timeout=10).stdout.strip()
        return float(out) if out and out != "N/A" else 0.0
    except Exception:
        return 0.0


class PyAVSampler:
    def __init__(self, path, time_of, index_of, keep_full=False, full_w=1920):
        self.full_w = full_w
        self.c = av.open(path)
        self.s = self.c.streams.video[0]
        self.s.thread_type = "SLICE"         # no frame-threading latency; parallel samplers instead
        self.s.codec_context.skip_frame = "NONKEY"
        self.tb = float(self.s.time_base)
        self.start = self.s.start_time or 0
        self.time_of, self.index_of, self.keep_full = time_of, index_of, keep_full
        try:
            self.rot = int(self.s.metadata.get("rotate", 0)) % 360
        except (ValueError, TypeError):
            self.rot = 0

    def sample(self, slot):
        target = int(self.time_of(slot) / self.tb) + self.start
        self.c.seek(target, stream=self.s, backward=True, any_frame=False)
        frame, sizes = None, []
        for pkt in self.c.demux(self.s):
            if frame is None:
                for f in pkt.decode():          # the flush packet at EOF also releases delayed frames
                    frame = f
                    break
                if frame is None and pkt.size == 0:
                    break
            elif pkt.size:
                sizes.append(pkt.size)          # following (inter) packets: bits spent on change = motion
                if len(sizes) >= 6:
                    break
        if frame is None:
            return None
        t = ((frame.pts or 0) - self.start) * self.tb
        idx = max(0, self.index_of(t + 1e-4))
        w = min(640, frame.width)
        h = max(2, int(frame.height * w / frame.width) // 2 * 2)
        gray = frame.reformat(width=w, height=h, format="gray").to_ndarray()
        tw = THUMB_W
        thumb = frame.reformat(width=tw, height=max(2, int(frame.height * tw / frame.width)),
                               format="bgr24").to_ndarray()
        rot = self.rot or ((-int(getattr(frame, "rotation", 0) or 0)) % 360)
        if rot in _ROTATE:
            thumb = cv2.rotate(thumb, _ROTATE[rot])
            gray = cv2.rotate(gray, _ROTATE[rot])
        full = None
        if self.keep_full:
            fw = min(self.full_w, frame.width)
            full = frame.reformat(width=fw, height=max(2, int(frame.height * fw / frame.width) // 2 * 2),
                                  format="bgr24").to_ndarray()
            if rot in _ROTATE:
                full = cv2.rotate(full, _ROTATE[rot])
        motion = float(np.mean(sizes)) / 1024 if sizes else 0.0
        return _Result(slot, idx, thumb, sharpness(gray), motion, _hist(thumb), full)

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass


def default_workers(width=1920, height=1080, samples=300):
    """Enough parallel decoders for ~1-2 s - sized to the video and to what the PC can spare right now."""
    from AIO_QC.core.procs import plan_analysis_workers
    return plan_analysis_workers(width, height, samples)


class Analyzer:
    """Samples `grid` with parallel decoders. on_result(_Result) is called from worker threads.
    request(idx) moves the nearest pending slot to the front (hover)."""

    def __init__(self, path, grid, time_of, index_of, workers=None, keep_full=False, use_pyav=True, ffprobe=None,
                 ffmpeg=None, size=(0, 0), prefer=None):
        """prefer: 'pyav' | 'ffmpeg' | 'opencv' | None (best available)."""
        self.path, self.grid = path, list(grid)
        self.time_of, self.index_of = time_of, index_of
        self.workers = workers or default_workers(size[0] or 1920, size[1] or 1080, len(self.grid))
        self.keep_full = keep_full
        self.ffmpeg, self.size = ffmpeg, size
        order = [k for k in ("pyav", "ffmpeg", "opencv")
                 if (k != "pyav" or (use_pyav and HAVE_PYAV)) and (k != "ffmpeg" or (ffmpeg and size[0]))]
        if prefer in order:
            order.remove(prefer)
            order.insert(0, prefer)
        self.kinds = order
        self.kind = order[0]
        self.gop = None
        self.start_time = 0.0
        self._ffprobe = ffprobe
        self._fails = 0
        if self.kind == "ffmpeg":
            self.start_time = probe_start_time(path, ffprobe)
        if self.kind == "opencv":
            self.gop, _ = estimate_gop(path, ffprobe)
        self._q = deque(self.grid[i] for i in coarse_to_fine(len(self.grid)))
        self._done = set()
        self._prio = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.throttle = False

    @property
    def backend(self):
        return self.kind if self.kind != "opencv" else f"opencv (gop {self.gop or '?'})"

    def _fallback(self):
        """The current sampler keeps failing on this file: move to the next kind."""
        with self._lock:
            i = self.kinds.index(self.kind)
            if i + 1 >= len(self.kinds):
                return False
            self.kind = self.kinds[i + 1]
            self._fails = 0
            if self.kind == "ffmpeg":
                self.start_time = probe_start_time(self.path, self._ffprobe)
            if self.kind == "opencv" and self.gop is None:
                self.gop, _ = estimate_gop(self.path, self._ffprobe)
            return True

    def request(self, idx):
        import bisect
        i = bisect.bisect_left(self.grid, idx)
        cands = [self.grid[j] for j in (i - 1, i) if 0 <= j < len(self.grid)]
        if cands:
            g = min(cands, key=lambda x: abs(x - idx))
            with self._lock:
                if g not in self._done:
                    self._prio = g

    def stop(self):
        self._stop.set()

    def _next(self):
        with self._lock:
            if self._prio is not None and self._prio not in self._done:
                p, self._prio = self._prio, None
                self._done.add(p)
                return p, True
            while self._q:
                g = self._q.popleft()
                if g not in self._done:
                    self._done.add(g)
                    return g, False
        return None, False

    def _make(self):
        if self.kind == "pyav":
            return PyAVSampler(self.path, self.time_of, self.index_of, self.keep_full)
        if self.kind == "ffmpeg":
            return FfmpegSampler(self.ffmpeg, self.path, self.size[0], self.size[1], self.time_of, self.index_of,
                                 self.keep_full, self.start_time)
        return OpenCVSampler(self.path, self.gop, self.keep_full)

    def run(self, on_result):
        """Blocks until everything is sampled (or stopped)."""
        def worker():
            kind, smp = None, None
            try:
                while not self._stop.is_set():
                    if kind != self.kind:           # (re)create after a fallback
                        if smp is not None:
                            smp.close()
                        kind = self.kind
                        try:
                            smp = self._make()
                        except Exception:
                            if not self._fallback():
                                return
                            continue
                    slot, urgent = self._next()
                    if slot is None:
                        return
                    try:
                        r = smp.sample(slot)
                    except Exception:
                        r = None
                    if r is not None:
                        self._fails = 0
                        on_result(r, urgent)
                    else:
                        self._fails += 1
                        if self._fails >= 3 and self._fallback():
                            with self._lock:             # retry this slot with the next sampler kind
                                self._done.discard(slot)
                                self._q.appendleft(slot)
                    if self.throttle:
                        time.sleep(0.04)
            finally:
                if smp is not None:
                    smp.close()
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
