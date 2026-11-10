# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Video sources with a common, frame-index based interface.

  OpenCVSource  - cv2.VideoCapture (FFmpeg). Supports HW decoding. Assumes a
                  constant frame rate (OpenCV maps frame index <-> time via fps).
  PyAVSource    - PyAV (optional dependency, `pip install av`). Builds an exact
                  per-frame timestamp index by demuxing packets (no decoding), so
                  frame indices and times are exact for variable-frame-rate
                  (phone/screen-recorded) video. Applies rotation metadata.

open_source(path, backend="auto") picks PyAV only when it matters (VFR or
rotated video) and PyAV is installed; otherwise OpenCV (with HW decode).

Interface:  width, height, fps, frame_count, timestamps (list[float] | None),
            is_vfr, rotation, backend, hw_accel,
            read() -> (ok, bgr), grab() -> ok, seek(idx), release(),
            time_of(idx), index_of(seconds)
"""
import bisect
import os

import cv2
import numpy as np

try:
    import av  # optional
    HAVE_PYAV = True
    # PyAV's Python log callback is invoked from FFmpeg's decoder threads and must take the GIL; with several
    # decoders running at once (playback + analysis + previews) that is a known cause of hangs / crashes.
    # Hand logging back to FFmpeg's own C callback and keep it quiet.
    try:
        av.logging.restore_default_callback()
    except Exception:
        pass
    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass
except Exception:  # pragma: no cover
    av = None
    HAVE_PYAV = False

_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class _Timebase:
    """Frame index <-> seconds, exact when per-frame timestamps are known."""
    timestamps = None
    fps = 30.0
    frame_count = 0

    def time_of(self, idx):
        if self.timestamps:
            idx = max(0, min(int(idx), len(self.timestamps) - 1))
            return self.timestamps[idx]
        return idx / self.fps if self.fps else 0.0

    def index_of(self, seconds):
        """Index of the frame on screen at `seconds`."""
        if self.timestamps:
            return max(0, bisect.bisect_right(self.timestamps, seconds + 1e-6) - 1)
        idx = int(seconds * self.fps + 1e-6)
        return max(0, min(idx, self.frame_count - 1)) if self.frame_count else idx

    def duration(self):
        if self.timestamps and len(self.timestamps) > 1:
            last = self.timestamps[-1]
            return last + (last - self.timestamps[-2])
        return self.frame_count / self.fps if self.fps else 0.0


class OpenCVSource(_Timebase):
    backend = "opencv"

    def __init__(self, path, hw_accel=True):
        self.path = path
        self.cap, self.hw_accel = self._open(path, hw_accel)
        if not self.cap.isOpened():
            raise IOError(f"OpenCV could not open {path}")
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # rotate phone video upright
        self.rotation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360 \
            if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 0 < fps < 1000 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Real (post-rotation) size comes from an actual decoded frame; the
        # WIDTH/HEIGHT properties can report the pre-rotation size.
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f"Could not decode first frame of {path}")
        self.height, self.width = frame.shape[:2]
        self._pos0 = (self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0   # OpenCV clock of frame 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timestamps = None
        self.keyframes = None
        self.is_vfr = False
        self._pending = False          # a frame was grabbed by an exact seek and not yet returned
        # Exact per-frame timing from the MP4/MOV index (milliseconds, no decoding). OpenCV alone assumes a
        # constant frame rate, which maps phone (VFR) videos' frame numbers to the wrong times and frames.
        from AIO_QC.tools.frame_extractor.core.mp4index import mp4_index
        idx = mp4_index(path)
        if idx is not None and (self.frame_count <= 0 or abs(len(idx[0]) - self.frame_count) <= 2):
            ts, keys = idx
            self.timestamps, self.keyframes = ts, keys
            self.frame_count = len(ts)
            self.is_vfr = _looks_vfr(ts)
            if ts[-1] > 0:
                self.fps = (len(ts) - 1) / ts[-1]

    @staticmethod
    def _open(path, hw_accel):
        if hw_accel and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
                if cap.isOpened():
                    ok, _ = cap.read()  # some HW decoders only fail on first read
                    if ok:
                        hw = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION)) != int(cv2.VIDEO_ACCELERATION_NONE)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        return cap, hw
                cap.release()
            except cv2.error:
                pass
        n = int(os.environ.get("FX_DECODE_THREADS", "0") or 0)    # set by procs.worker_init in batch workers
        if n > 0 and hasattr(cv2, "CAP_PROP_N_THREADS"):
            params = [cv2.CAP_PROP_N_THREADS, n]
            if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):       # explicitly CPU here too
                params += [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE]
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
            if cap.isOpened():
                return cap, False
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):   # explicitly CPU: never lands on the GPU decoder
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
            if cap.isOpened():
                return cap, False
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG), False

    def read(self):
        if self._pending:
            self._pending = False
            return self.cap.retrieve()
        return self.cap.read()

    def grab(self):
        if self._pending:
            self._pending = False
            return True
        return self.cap.grab()

    def current_index(self):
        """Index of the frame just grabbed/read, from its real timestamp."""
        t = (self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0 - self._pos0
        ts = self.timestamps
        i = bisect.bisect_left(ts, t - 1e-4)
        cands = [j for j in (i - 1, i, i + 1) if 0 <= j < len(ts)]
        return min(cands, key=lambda j: abs(ts[j] - t))

    def seek(self, idx):
        self._pending = False
        if not self.timestamps:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            return
        self._seek_exact(max(0, min(int(idx), self.frame_count - 1)))

    def _seek_exact(self, idx):
        """Seek by time a little early, then step to the frame whose real timestamp is ts[idx].
        Afterwards the next read()/grab() returns exactly frame idx (it is held as 'pending')."""
        ts = self.timestamps
        target = ts[idx]
        gap = (ts[min(idx + 1, len(ts) - 1)] - ts[max(idx - 1, 0)]) or 0.05
        margin = 2 * gap
        for _ in range(5):
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (target - margin + self._pos0) * 1000.0))
            if not self.cap.grab():
                margin = margin * 3 + 0.5
                continue
            cur = self.current_index()
            if cur > idx:                         # landed after the target: seek earlier
                margin = margin * 3 + 0.5
                continue
            steps = 0
            while cur < idx and steps < 20000:
                if not self.cap.grab():
                    break
                cur = self.current_index()
                steps += 1
            if cur == idx:
                self._pending = True
                return
            margin = margin * 3 + 0.5
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)    # last resort

    def release(self):
        self.cap.release()


class PyAVSource(_Timebase):
    backend = "pyav"
    hw_accel = False

    def __init__(self, path):
        if not HAVE_PYAV:
            raise ImportError("PyAV is not installed (pip install av)")
        self.path = path
        pts, keys, tb, avg_rate, rotation = self._index(path)
        if not pts:
            raise IOError(f"No video packets found in {path}")
        self._pts = pts
        self._pts_to_idx = {p: i for i, p in enumerate(pts)}
        self._key_idx = sorted(self._pts_to_idx[p] for p in keys) or [0]
        t0 = pts[0]
        self.timestamps = [(p - t0) * tb for p in pts]
        self.frame_count = len(pts)
        dur = self.timestamps[-1] if self.frame_count > 1 else 0
        self.fps = avg_rate or ((self.frame_count - 1) / dur if dur > 0 else 30.0)
        self.is_vfr = _looks_vfr(self.timestamps)
        self.keyframes = list(self._key_idx)
        self.rotation = rotation

        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"   # frame+slice threading
        self._decoder = self.container.decode(self.stream)
        self._peek = None                  # decoded frame waiting to be returned
        self._frame_rot = None             # rotation read from frames (PyAV >= 14)
        ok, frame = self.read()
        if not ok:
            raise IOError(f"Could not decode first frame of {path}")
        self.height, self.width = frame.shape[:2]
        self.seek(0)

    @staticmethod
    def _index(path):
        """Demux (not decode) all packets: fast exact per-frame timestamps."""
        with av.open(path) as c:
            s = c.streams.video[0]
            tb = float(s.time_base)
            rate = float(s.average_rate) if s.average_rate else None
            rotation = 0
            try:
                rotation = int(s.metadata.get("rotate", 0)) % 360  # old FFmpeg tag (clockwise)
            except (ValueError, TypeError):
                pass
            pts, keys = [], set()
            for pkt in c.demux(s):
                if pkt.pts is None:
                    continue
                pts.append(pkt.pts)
                if pkt.is_keyframe:
                    keys.add(pkt.pts)
            pts = sorted(set(pts))
        return pts, keys, tb, rate, rotation

    def _next_frame(self):
        if self._peek is not None:
            f, self._peek = self._peek, None
            return f
        try:
            return next(self._decoder)
        except (StopIteration, av.error.EOFError):
            return None

    def _to_bgr(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rot = self.rotation
        if self._frame_rot is None:
            # PyAV >= 14 exposes the display-matrix rotation (counter-clockwise)
            r = getattr(frame, "rotation", None)
            self._frame_rot = (-int(r)) % 360 if r else 0
        if self._frame_rot:
            rot = self._frame_rot
        if rot in _ROTATE:
            img = cv2.rotate(img, _ROTATE[rot])
        return img

    def read(self):
        f = self._next_frame()
        if f is None:
            return False, None
        return True, self._to_bgr(f)

    def grab(self):
        return self._next_frame() is not None

    def seek(self, idx):
        idx = max(0, min(int(idx), self.frame_count - 1))
        target = self._pts[idx]
        k = bisect.bisect_right(self._key_idx, idx) - 1
        key_pts = self._pts[self._key_idx[max(0, k)]]
        self.container.seek(key_pts, stream=self.stream, backward=True, any_frame=False)
        self._decoder = self.container.decode(self.stream)
        self._peek = None
        while True:  # decode forward to the exact frame
            try:
                f = next(self._decoder)
            except (StopIteration, av.error.EOFError):
                return
            if f.pts is None or f.pts >= target:
                self._peek = f
                return

    def release(self):
        try:
            self.container.close()
        except Exception:
            pass


def _looks_vfr(ts, tol=0.02):
    if len(ts) < 3:
        return False
    d = np.diff(np.asarray(ts[: min(len(ts), 5000)]))
    d = d[d > 0]
    if d.size < 2:
        return False
    med = float(np.median(d))
    return bool(np.mean(np.abs(d - med) > tol * med + 1e-4) > 0.05)


def _quick_probe(path, n=300):
    """Cheap PyAV peek at the first packets: (is_vfr, rotated)."""
    with av.open(path) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        rotated = bool(s.metadata.get("rotate", "0") not in ("0", ""))
        pts = []
        for pkt in c.demux(s):
            if pkt.pts is not None:
                pts.append(pkt.pts)
            if len(pts) >= n:
                break
        pts.sort()
        vfr = _looks_vfr([p * tb for p in pts])
        if not rotated:
            try:  # PyAV >= 14: rotation lives on decoded frames
                c.seek(0)
                f = next(c.decode(s))
                rotated = bool(getattr(f, "rotation", 0))
            except Exception:
                pass
    return vfr, rotated


def open_source(path, backend="auto", hw_accel=True):
    """backend: 'auto' | 'opencv' | 'pyav'."""
    if backend == "pyav":
        return PyAVSource(path)
    if backend == "auto" and HAVE_PYAV:
        try:
            vfr, rotated = _quick_probe(path)
            if vfr or rotated:
                return PyAVSource(path)
        except Exception:
            pass  # fall through to OpenCV
    return OpenCVSource(path, hw_accel=hw_accel)


def format_time(seconds):
    """HH:MM:SS.mmm"""
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_time(text):
    """Accepts HH:MM:SS(.fff), MM:SS(.fff) or plain seconds. Returns seconds or raises ValueError."""
    text = text.strip()
    if not text:
        raise ValueError("empty time")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid time '{text}'")
    secs = 0.0
    for p in parts:
        v = float(p)
        if v < 0:
            raise ValueError(f"invalid time '{text}'")
        secs = secs * 60 + v
    return secs


# ---------------------------------------------------------------------------
# Keyframe access (PyAV): fast scrubbing previews and timeline thumbnails.
# ---------------------------------------------------------------------------
class KeyframeIndex:
    """Demux-only scan (no decoding): per-frame pts, keyframe positions and
    packet sizes. Packet size of inter frames is what the encoder spent on
    change between frames, so it is an almost free motion signal."""

    def __init__(self, path):
        if not HAVE_PYAV:
            raise ImportError("PyAV is not installed (pip install av)")
        self.path = path
        with av.open(path) as c:
            s = c.streams.video[0]
            self.time_base = float(s.time_base)
            try:
                self.rotation = int(s.metadata.get("rotate", 0)) % 360
            except (ValueError, TypeError):
                self.rotation = 0
            sizes, keys = {}, set()
            for pkt in c.demux(s):
                if pkt.pts is None:
                    continue
                sizes[pkt.pts] = pkt.size
                if pkt.is_keyframe:
                    keys.add(pkt.pts)
        self.pts = sorted(sizes)
        self.frame_count = len(self.pts)
        pos = {p: i for i, p in enumerate(self.pts)}
        self.keyframes = sorted(pos[p] for p in keys)
        self.sizes = np.array([sizes[p] for p in self.pts], np.float32)

    def keyframe_before(self, idx):
        k = bisect.bisect_right(self.keyframes, idx) - 1
        return self.keyframes[max(0, k)] if self.keyframes else 0

    def motion(self, lo, hi):
        """Mean inter-frame packet size (kB) over [lo, hi): relative motion."""
        seg = self.sizes[lo:hi]
        keyset = set(self.keyframes)
        inter = [v for i, v in enumerate(seg, lo) if i not in keyset]
        return float(np.mean(inter)) / 1024 if inter else 0.0


class KeyframeReader:
    """Decode single keyframes quickly (seek + one decode, non-key frames skipped)."""

    def __init__(self, index):
        self.index = index
        self.container = av.open(index.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        try:
            self.stream.codec_context.skip_frame = "NONKEY"
        except Exception:
            pass
        self._frame_rot = None

    def read(self, kidx):
        """BGR image of keyframe number kidx (a frame index that is a keyframe)."""
        target = self.index.pts[kidx]
        self.container.seek(target, stream=self.stream, backward=True, any_frame=False)
        for f in self.container.decode(self.stream):
            if f.pts is None or f.pts >= target or f.key_frame:
                img = f.to_ndarray(format="bgr24")
                rot = self.index.rotation
                if self._frame_rot is None:
                    r = getattr(f, "rotation", None)
                    self._frame_rot = (-int(r)) % 360 if r else 0
                rot = self._frame_rot or rot
                return cv2.rotate(img, _ROTATE[rot]) if rot in _ROTATE else img
        return None

    def release(self):
        try:
            self.container.close()
        except Exception:
            pass
