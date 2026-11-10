# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
VideoEngine - media-player style decoding for FrameExtractor.

Design (same idea as mpv/VLC, simplified):
  * ONE thread owns the cv2.VideoCapture. The GUI never touches it, so there is
    no blocking decode on the GUI thread and no fighting over the decoder position.
  * The decoder keeps a read-ahead buffer ("ahead") of frames, each holding
      - the full-resolution BGR array (for exact saving), and
      - a QImage already downscaled to the canvas size (for cheap painting).
    Buffer size is derived from a memory budget, so 4K/8K don't eat RAM.
  * Playback is driven by a wall clock: the GUI asks "which frame is due now?"
    and late frames are dropped instead of slowing the video down. If the decoder
    itself is behind, it grab()s (decodes without colour conversion/copy) frames
    that are already late.
  * Pausing keeps the decoder positioned and keeps prefetching, so resume and
    single-frame forward steps are instant. A small history of shown frames makes
    short backward steps instant too.
  * Decoding goes through core.sources: OpenCV (HW decode) or PyAV (exact
    timestamps for variable-frame-rate video, rotation metadata).
  * Seek requests are coalesced (latest wins), which keeps scrubbing responsive
    even on long-GOP 4K footage.
"""
import time
import threading
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import bisect
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from AIO_QC.tools.frame_extractor.core.sources import open_source


@dataclass
class Frame:
    idx: int
    bgr: np.ndarray   # full-resolution decoded frame (saving/cropping); None for real-time playback frames
    image: QImage     # display-resolution image (used for painting)
    preview: bool = False   # keyframe preview while scrubbing (not the exact frame yet)
    small: np.ndarray = None  # display-size BGR (real-time pipeline frames have no full-res bgr)
    nv12: tuple = None        # (Y, UV) full-resolution planes from the pipeline (GPU canvas: original quality)
    t_ready: float = 0.0      # when it entered the read-ahead queue (drop diagnosis: late decode vs missed refresh)


def nv12_to_bgr(y, uv, space="601", full_range=False):
    """Full-resolution BGR from NV12 planes with the video's own matrix - the same colours the exact decoder
    (OpenCV follows the file's tag, BT.601 when untagged) produces, so a paused frame never shifts colour."""
    h, w = y.shape[:2]
    if space == "601" and not full_range:
        buf = np.vstack([y, uv.reshape(h // 2, w)])
        return cv2.cvtColor(np.ascontiguousarray(buf), cv2.COLOR_YUV2BGR_NV12)
    from AIO_QC.tools.frame_extractor.widgets.gl_shaders import yuv_uniforms
    r, g, b, y0 = yuv_uniforms(space, full_range)
    yf = y.astype(np.float32) * (1 / 255.0) - y0
    u = cv2.resize(uv[..., 0], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * (1 / 255.0) - 0.5
    v = cv2.resize(uv[..., 1], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * (1 / 255.0) - 0.5
    out = np.empty((h, w, 3), np.uint8)
    for ch, row in ((2, r), (1, g), (0, b)):
        out[..., ch] = np.clip((row[0] * yf + row[1] * u + row[2] * v) * 255.0 + 0.5, 0, 255)
    return out


def nv12_small_bgr(y, uv, max_w=1280):
    """Cheap BGR preview of an NV12 frame (decimate first, convert the small image)."""
    h, w = y.shape[:2]
    k = 1
    while w // k > max_w:
        k *= 2
    ys = y[::k, ::k]
    uvs = uv[::k, ::k] if k > 1 else uv
    hs, ws = (ys.shape[0] // 2) * 2, (ys.shape[1] // 2) * 2
    ys, uvs = ys[:hs, :ws], uvs[:hs // 2, :ws // 2]
    buf = np.vstack([ys, uvs.reshape(hs // 2, ws)])
    return cv2.cvtColor(np.ascontiguousarray(buf), cv2.COLOR_YUV2BGR_NV12)


def fit_size(src_w, src_h, box_w, box_h):
    """Size that fits (src_w, src_h) inside the box, keeping aspect. Never upscales:
    upscaling small videos is left to the painter, which is free."""
    if box_w <= 0 or box_h <= 0:
        return src_w, src_h
    scale = min(box_w / src_w, box_h / src_h, 1.0)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def bgr_to_display(bgr, box, interp=cv2.INTER_AREA):
    """Downscale in OpenCV (fast, multithreaded) and wrap as QImage without a
    BGR->RGB pass (Qt >= 5.14 understands BGR888 directly)."""
    h, w = bgr.shape[:2]
    tw, th = fit_size(w, h, *box)
    if (tw, th) != (w, h):
        bgr = cv2.resize(bgr, (tw, th), interpolation=interp)
    bgr = np.ascontiguousarray(bgr)
    return QImage(bgr.data, tw, th, bgr.strides[0], QImage.Format.Format_BGR888).copy()


class VideoEngine(QThread):
    # Emitted (from the decoder thread) with a Frame produced by a seek/step
    # request that could not be served from cache.
    still_ready = pyqtSignal(object)
    # Emitted when EOF reveals the container over-reported its length:
    # the new (true) frame count.
    length_changed = pyqtSignal(int)

    RESYNC_BEHIND_S = 0.4     # decoder this far behind the clock -> jump ahead (keeps real time)
    RESYNC_LEAD_S = 0.25      # ...to slightly ahead of the clock, to absorb the seek / restart
    PREFETCH_DELAY = 0.15   # s of seek-idle before prefetching again (keeps scrubbing snappy)
    MAX_CONSEC_DROPS = 5    # always render at least 1 of every N+1 frames while behind
    LATE_FRAMES = 3         # exact decoder skips colour conversion only when this far behind (not on jitter)
    PREBUFFER_S = 0.20      # decoded ahead before the clock starts (no stutter at play)
    RT_BUDGET = 640 * 2 ** 20   # bytes of read-ahead for pipeline frames

    PAUSED_AHEAD = 8          # frames prefetched while paused (instant stepping without a RAM spike)

    def __init__(self, path, display_size=(1280, 720), hw_accel=False, cache_mb=768, backend="auto", src=None):
        """src: an already opened core.sources source (opened off the UI thread by the app)."""
        super().__init__()
        self.path = path
        self.src = src if src is not None else open_source(path, backend=backend, hw_accel=hw_accel)
        self.hw_accel = self.src.hw_accel
        self.backend = self.src.backend
        self.fps = self.src.fps
        self.frame_count = self.src.frame_count
        self.width, self.height = self.src.width, self.src.height
        self.timestamps = self.src.timestamps      # exact per-frame times (PyAV) or None (CFR)
        self.is_vfr = self.src.is_vfr
        self.rotation = self.src.rotation
        # Decode forward with grab() instead of seeking when the target is this
        # close ahead. A seek decodes from the previous keyframe anyway, so for
        # small forward jumps plain decoding is never slower.
        self.seek_grab_limit = max(15, int(self.fps))

        # Memory-budgeted caches (full-res frames dominate memory).
        # each cached frame = full-res BGR + its display copy
        dw, dh = fit_size(self.width, self.height, *display_size)
        frame_bytes = max(1, self.width * self.height * 3 + dw * dh * 3)
        slots = max(6, min(int(cache_mb * 1024 * 1024 // frame_bytes), 48))
        self.max_ahead = max(3, slots * 2 // 3)
        self.max_history = max(2, slots - self.max_ahead)

        self._cond = threading.Condition()
        self._ahead = deque()                         # decoded, not yet shown
        self._history = deque(maxlen=self.max_history)  # recently shown
        self._pending = None      # (idx, emit_still) latest seek request
        self._pending_fast = None # latest keyframe-preview request (scrubbing)
        self._pending_exact = None  # exact full-res frame for the paused pipeline frame (stream untouched)
        self.colorspace = ("601", False)   # set by the app from the file's tags (NV12 -> BGR conversion)
        self.display_hz = 0.0     # display refresh rate (display sync)
        self.display_sync = True  # lock frames to the refresh when rates are within 1 % (like mpv display-resample)
        self._sync = 1.0          # clock rate factor from display sync
        self._kf_index = None     # core.sources.KeyframeIndex (PyAV) when available
        self._kf_reader = None
        self._pv = None           # (kind, ffmpeg) keyframe previews by seeking - no file index needed
        self._pv_obj = None
        self._rt = None           # real-time pipeline config: dict(ffmpeg, chain, version) or None
        self._pipe_mode = False   # this play session uses the pipeline
        self._pipe = None
        self._pipe_gen = -1
        self._pipe_lock = threading.Lock()
        self._last_shown = 0
        self.rt_active = False    # a pipeline frame was shown in this session (for the status bar)
        self._gen = 0             # bumps on every reposition; stale decodes are discarded
        self._stream_next = 0     # index the next frame appended to _ahead will have
        self._eof = False
        self._dead = False        # decoder thread exited (error) -> treat as end
        self._quit = False
        self._playing = False
        self._speed = 1.0
        self._clock_t0 = None     # wall time at which _clock_f0 was shown
        self._clock_f0 = 0
        self._prefetch_after = 0.0
        self._display_size = tuple(display_size)
        self._decode_pos = 0      # decoder-thread only: index the next read() returns
        self.dropped = 0
        # playback statistics (Playback stats overlay): where do frames go?
        self.display_images = True   # False with the GPU canvas: it uploads full-res frames, no display copy needed
        self.presented = 0           # frames handed to the screen
        self.drop_present = 0        # decoded in time, but its display slot passed (UI missed a refresh)
        self.drop_decode = 0         # decoder behind: decoded without conversion / never shown
        self.drop_resync = 0         # skipped by jumping ahead to stay in real time
        self.drop_source = 0         # the video has frames closer together than any display refresh (unavoidable)
        self.decode_ms = 0.0         # EMA of decode + convert per frame (decoder thread)
        self.resyncs = 0
        self._last_resync = 0.0
        self._keyframes = getattr(self.src, "keyframes", None)  # from the MP4 index / PyAV (resync without preroll)
        self._restart_s = self.RESYNC_LEAD_S   # measured time from (re)start to first frame (EMA)
        self._restart_t0 = None

        logger.info(
            f"Engine: {self.width}x{self.height} @ {self.fps:.3f} fps, "
            f"{self.frame_count} frames, backend: {self.backend}, HW decode: {self.hw_accel}, "
            f"VFR: {self.is_vfr}, rotation: {self.rotation}, "
            f"cache: {self.max_ahead} ahead / {self.max_history} history"
        )

    # ------------------------------------------------------------------ GUI API
    @property
    def is_playing(self):
        return self._playing

    @property
    def speed(self):
        return self._speed

    def set_display_size(self, w, h):
        self._display_size = (int(w), int(h))

    def make_display(self, frame):
        """Re-render a frame for the current display size (e.g. after a resize)."""
        if frame.bgr is not None:
            frame.image = bgr_to_display(frame.bgr, self._display_size, cv2.INTER_AREA)
        return frame

    # ----------------------------------------------------------- real-time pipeline
    def set_realtime(self, cfg):
        """cfg = dict(ffmpeg=..., chain=..., version=(maj, min)) enables the display-size playback
        pipeline (core.preview_pipe); None = always use the exact decoder."""
        self._rt = dict(cfg) if cfg else None

    @property
    def realtime(self):
        return self._rt

    def _close_pipe(self):
        with self._pipe_lock:
            pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.close()

    def time_of(self, idx):
        return self.src.time_of(idx)

    def index_of(self, seconds):
        return self.src.index_of(seconds)

    def duration(self):
        return self.src.duration()

    def clamp(self, idx):
        hi = self.frame_count - 1 if self.frame_count > 0 else idx
        return max(0, min(int(idx), hi))

    def set_keyframe_index(self, index):
        """Enable instant keyframe previews for scrubbing (PyAV index built by
        the timeline worker). Ignored if its frame numbering disagrees with ours."""
        if index is not None and self.frame_count and abs(index.frame_count - self.frame_count) > 2:
            logger.info("Keyframe index length differs from the decoder's; fast scrubbing disabled")
            return
        self._kf_index = index
        if index is not None and not self._keyframes:
            self._keyframes = list(index.keyframes)

    def set_keyframes(self, keyframes, frame_count=None):
        """Keyframe positions (from PyAV or ffprobe). Ignored if the numbering disagrees with ours."""
        if frame_count and self.frame_count and abs(frame_count - self.frame_count) > 2:
            return
        self._keyframes = sorted(keyframes) if keyframes else None

    def set_preview_sampler(self, kind, ffmpeg=None):
        """kind 'pyav' | 'ffmpeg': instant keyframe previews while scrubbing (seek + one keyframe decode)."""
        self._pv = (kind, ffmpeg) if kind else None

    @property
    def fast_scrub(self):
        return self._kf_index is not None or self._pv is not None

    def request_frame(self, idx, fast=False):
        """Ask for frame `idx` while paused/scrubbing. Returns the Frame right away
        if cached; otherwise schedules a decode and returns None (the result
        arrives through `still_ready`). fast=True (while dragging) shows the
        nearest preceding keyframe immediately instead of decoding the exact frame."""
        idx = self.clamp(idx)
        fast = fast and (self._kf_index is not None or self._pv is not None)
        with self._cond:
            frame = self._take_cached(idx) if not fast else next(
                (f for f in reversed(self._history) if f.idx == idx), None)
            if frame is not None:
                self._cond.notify_all()
                return frame
            if fast:
                self._pending_fast = idx
                self._cond.notify_all()
                return None
            self._reposition(idx, emit=True)
            self._prefetch_after = time.monotonic() + self.PREFETCH_DELAY
            self._cond.notify_all()
        return None

    def play(self, start_idx):
        """Start the clock so that `start_idx` is the first frame shown."""
        start_idx = self.clamp(start_idx)
        if self._rt:
            with self._cond:
                # paused pipeline that already continues at start_idx: resume instantly, like a media player
                alive = self._pipe is not None and self._pipe_gen == self._gen
                if self._pipe_mode and ((self._ahead and self._ahead[0].idx == start_idx) or
                                        (not self._ahead and alive and self._stream_next == start_idx)):
                    self._playing = True
                    self._clock_t0 = None
                    self._clock_f0 = start_idx
                    self._cond.notify_all()
                    return
                if self._pipe_mode and any(f.idx == start_idx for f in self._ahead):
                    while self._ahead[0].idx != start_idx:
                        self._ahead.popleft()
                    self._playing = True
                    self._clock_t0 = None
                    self._clock_f0 = start_idx
                    self._cond.notify_all()
                    return
                self._ahead.clear()
                self._gen += 1
                self._eof = False
                self._pending = None
                self._pipe_mode = True
                self._stream_next = start_idx
                self._playing = True
                self._clock_t0 = None
                self._clock_f0 = start_idx
                self._cond.notify_all()
            return
        with self._cond:
            if any(f.idx == start_idx for f in self._ahead):
                while self._ahead[0].idx != start_idx:
                    self._ahead.popleft()
            elif not (not self._ahead and self._pending is None and self._stream_next == start_idx):
                self._reposition(start_idx, emit=False)
            self._playing = True
            self._clock_t0 = None       # starts when the first frame is actually due
            self._clock_f0 = start_idx
            self._cond.notify_all()

    def pause(self):
        """Stops the clock. A playing pipeline stays alive with its decoded frames (ffmpeg simply blocks on
        the full pipe), so resuming is instant - no new process, no keyframe re-decode (long-GOP CCTV)."""
        with self._cond:
            self._playing = False
            self._clock_t0 = None
            self._cond.notify_all()

    def fetch_exact(self, idx):
        """Exact full-resolution frame for display / saving while the pipeline is paused - decoded by the exact
        decoder without disturbing the pipeline's stream. Arrives through still_ready."""
        with self._cond:
            f = next((x for x in reversed(self._history) if x.idx == idx and x.bgr is not None), None)
            if f is not None:
                return f
            self._pending_exact = int(idx)
            self._cond.notify_all()
        return None

    def exact_from_nv12(self, frame):
        """Full-res BGR for a pipeline NV12 frame (they are full resolution already): instant, no decode."""
        if frame.nv12 is None:
            return None
        bgr = nv12_to_bgr(*frame.nv12, *self.colorspace)
        disp = bgr_to_display(bgr, self._display_size, cv2.INTER_AREA) if self.display_images else None
        f = Frame(frame.idx, bgr, disp)
        with self._cond:
            self._history.append(f)
        return f

    def set_display_rate(self, hz):
        self.display_hz = float(hz or 0.0)
        self._update_sync()

    def _update_sync(self):
        """Display sync: with no audio to stay in step with, a video whose rate is within 1 % of the refresh
        (or of a whole fraction of it - 29.97 on 60 Hz, 23.976 on 48 Hz...) is played at exactly that rate, so
        every frame lands on its own refresh: no periodic repeat / skip (59.94 on 60 Hz drifts by one frame
        every ~17 s otherwise). The speed difference (0.1 %) is invisible."""
        self._sync = 1.0
        hz = self.display_hz
        rate = self.fps * self._speed
        if not (self.display_sync and hz > 20 and rate > 0):
            return
        for k in (1, 2, 3, 4, 5):
            target = hz / k
            if abs(rate - target) / target < 0.01:
                self._sync = target / rate
                return

    @property
    def sync_factor(self):
        return self._sync

    def set_speed(self, speed):
        with self._cond:
            if self._playing and self._clock_t0 is not None:
                now = time.monotonic()
                self._clock_f0 = self._clock_target(now)
                self._clock_t0 = now
            self._speed = float(speed)
        self._update_sync()

    def pop_due_frame(self, present_in=0.0, period=0.0):
        """Called once per display refresh (GPU canvas) or timer tick (software canvas).
        present_in: seconds until what is drawn now becomes visible (next vsync).
        period: the display's refresh period. Frames are matched to the refresh *nearest* their timestamp
        (within +-period/2, like mpv) rather than the last one before it - otherwise a 60 fps video on a 60 Hz
        screen, nearly in phase, skips frames on tiny timing noise.
        Returns the frame due at that refresh, or None."""
        with self._cond:
            if not self._playing or not self._ahead:
                return None
            present_at = time.monotonic() + present_in      # when what we draw now becomes visible
            now = present_at + period / 2.0                  # match frames to the nearest refresh
            if self._clock_t0 is None:
                # prebuffer a little before starting the clock, like a media player
                need = max(2, int(self.PREBUFFER_S * self.fps * max(1.0, self._speed)))
                need = min(need, self.RT_AHEAD if self._pipe_mode else self.max_ahead)
                if len(self._ahead) < need and not self._eof and not self._dead:
                    return None
                frame = self._ahead.popleft()
                # anchor the clock to the refresh the first frame appears on - NOT the window edge: anchored
                # at the edge, every later frame falls exactly on a window boundary and timing noise decides
                # (two frames in one refresh -> a skip)
                self._clock_t0, self._clock_f0 = present_at, frame.idx
            else:
                target = self._clock_target(now)
                frame = None
                while self._ahead and self._ahead[0].idx <= target:
                    if frame is not None:
                        self.dropped += 1
                        # skipped - why? next frame due < 10 ms later: source too dense for any display;
                        # arrived after it was due: decoder late; otherwise the UI missed a refresh
                        due = self._due_time(frame.idx)
                        if self._due_time(self._ahead[0].idx) - due < 0.010:
                            self.drop_source += 1
                        elif frame.t_ready and frame.t_ready > due + 0.002:
                            self.drop_decode += 1
                        else:
                            self.drop_present += 1
                    frame = self._ahead.popleft()
                if frame is None:
                    return None
            self._history.append(frame)
            self.presented += 1
            self._last_shown = frame.idx
            self._cond.notify_all()
            return frame

    def stats(self):
        with self._cond:
            q = len(self._ahead)
        return dict(presented=self.presented, drop_present=self.drop_present, drop_decode=self.drop_decode,
                    drop_source=self.drop_source,
                    drop_resync=self.drop_resync, queue=q, decode_ms=self.decode_ms,
                    pipeline=bool(self._pipe_mode), out=(self._rt or {}).get("out"))

    def reset_stats(self):
        self.presented = self.drop_present = self.drop_decode = self.drop_resync = self.drop_source = 0

    def at_end(self):
        with self._cond:
            if self._dead:
                return not self._ahead
            return self._eof and not self._ahead and self._pending is None

    def stop(self):
        with self._cond:
            self._quit = True
            self._cond.notify_all()
        self.wait()

    # ------------------------------------------------------- internals (locked)
    def _due_time(self, idx):
        """Wall-clock time at which frame idx should be on screen (current clock)."""
        if self._clock_t0 is None:
            return 0.0
        if self.timestamps:
            return self._clock_t0 + (self.timestamps[idx] - self.timestamps[self._clock_f0]) / (self._speed * self._sync)
        return self._clock_t0 + (idx - self._clock_f0) / (self.fps * self._speed * self._sync)

    def _clock_target(self, now):
        """Highest frame index that is due at wall time `now` (float for CFR)."""
        elapsed = (now - self._clock_t0) * self._speed * self._sync
        if self.timestamps:   # VFR-exact: walk the real timestamps
            t = self.timestamps[self._clock_f0] + elapsed
            return bisect.bisect_right(self.timestamps, t + 1e-6) - 1
        return self._clock_f0 + elapsed * self.fps

    def _reposition(self, idx, emit):
        if self._pipe_mode:
            self._pipe_mode = False          # a seek ends the paused pipeline; play() starts a fresh one
            self._pipe_gen = -1
        self._ahead.clear()
        self._gen += 1
        self._eof = False
        self._pending = (idx, emit)
        self._stream_next = idx + 1 if emit else idx

    def _take_cached(self, idx):
        for f in reversed(self._history):
            if f.idx == idx and f.bgr is not None:
                return f
        if any(f.idx == idx and f.bgr is not None for f in self._ahead):
            while True:
                f = self._ahead.popleft()
                self._history.append(f)   # skipped frames stay reachable for step-back
                if f.idx == idx:
                    return f
        return None

    # --------------------------------------------------------- decoder thread
    def _safe(self, fn, *a):
        """Decoder calls must never kill the thread: errors at a damaged tail
        (truncated files, HW decoder hiccups) are treated as end of stream."""
        try:
            return fn(*a)
        except Exception as e:
            if not getattr(self, "_warned", False):
                logger.warning(f"Decode error treated as end of stream: {e}")
                self._warned = True
            return None

    def _read(self):
        r = self._safe(self.src.read)
        return r if r is not None else (False, None)

    def _grab(self):
        return bool(self._safe(self.src.grab))

    def _found_end(self, idx):
        """Read failed at `idx`: frames 0..idx-1 are all there is."""
        if 0 < idx < self.frame_count or self.frame_count <= 0:
            logger.info(f"Container reported {self.frame_count} frames; actual length is {idx}")
            self.frame_count = idx
            self.length_changed.emit(idx)

    def _note_restart(self):
        if self._restart_t0 is not None:
            dt = time.monotonic() - self._restart_t0
            self._restart_s = 0.5 * self._restart_s + 0.5 * min(dt, 3.0)
            self._restart_t0 = None

    RT_AHEAD = 24     # upper bound; the byte budget decides for large (full-res NV12) frames

    def _rt_ahead(self):
        rt = self._rt or {}
        if rt.get("out") == "nv12":
            fb = self.width * self.height * 3 // 2
            return int(max(4, min(self.RT_AHEAD, self.RT_BUDGET // max(1, fb))))
        return self.RT_AHEAD

    def _fill_from_pipe(self, idx, gen, late, box, consec_drops):
        from AIO_QC.tools.frame_extractor.core.preview_pipe import PreviewPipe, display_size
        with self._pipe_lock:
            pipe = self._pipe if self._pipe_gen == gen else None
        if pipe is None:
            self._close_pipe()
            rt = self._rt
            if not rt:
                with self._cond:
                    self._pipe_mode = False
                return consec_drops
            native = rt.get("out") == "nv12"
            w, h = (self.width // 2 * 2, self.height // 2 * 2) if native else display_size(self.width, self.height, *box)
            t = self.time_of(idx)
            if idx > 0:   # land on the frame: start a quarter frame early (VFR-safe)
                t -= max(1e-4, (t - self.time_of(idx - 1)) / 4)
            try:
                if int(rt.get("parallel", 1)) > 1 and self._keyframes and self.timestamps:
                    from AIO_QC.tools.frame_extractor.core.preview_pipe import ParallelPipe
                    pipe = ParallelPipe(rt["ffmpeg"], self.path, rt["chain"], w, h, idx, self._keyframes,
                                        self.time_of, self.frame_count, int(rt["parallel"]),
                                        tuple(rt.get("version", (6, 0))), "nv12" if native else "bgr24")
                else:
                    pipe = PreviewPipe(rt["ffmpeg"], self.path, rt["chain"], w, h, max(0.0, t),
                                       version=tuple(rt.get("version", (6, 0))), pix_fmt="nv12" if native else "bgr24")
            except Exception as e:
                logger.warning(f"Real-time pipeline unavailable, using the exact decoder: {e}")
                self._rt = None
                with self._cond:
                    self._pipe_mode = False
                return consec_drops
            with self._pipe_lock:
                self._pipe, self._pipe_gen = pipe, gen
            self._pipe_start = idx
            if self._restart_t0 is None:
                self._restart_t0 = time.monotonic()
        small = pipe.read()
        if small is None:
            with self._cond:
                if gen != self._gen:
                    return consec_drops          # paused / seeked: the pipe was closed on purpose
            err = pipe.error
            self._close_pipe()
            if err or idx == getattr(self, "_pipe_start", idx):
                logger.warning(f"Real-time pipeline stopped at frame {idx} ({err or 'no frames'}); "
                               "continuing with the exact decoder")
                self._rt = None if idx == getattr(self, "_pipe_start", idx) else self._rt
                with self._cond:
                    if gen == self._gen:
                        self._pipe_mode = False   # the normal fill continues from idx
                return consec_drops
            with self._cond:
                if gen == self._gen:
                    if self.frame_count and idx < self.frame_count:
                        self._pipe_mode = False   # let the exact decoder finish / find the real end
                    else:
                        self._eof = True
            return consec_drops
        self._note_restart()
        # Pipeline frames are already decoded and scaled - skipping them saves nothing, so always
        # hand over the newest one; keyframe resync keeps the position on the clock.
        if pipe.pix_fmt == "nv12":
            y, uv = small
            frame, consec_drops = Frame(idx, None, None, nv12=(y, uv)), 0
        else:
            img = np.ascontiguousarray(small)
            q = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_BGR888).copy()
            frame, consec_drops = Frame(idx, None, q, small=img), 0
        with self._cond:
            if gen != self._gen:
                return consec_drops
            self._stream_next = idx + 1
            if frame is None:
                self.dropped += 1
            else:
                frame.t_ready = time.monotonic()
                self._ahead.append(frame)
                self.rt_active = True
        return consec_drops

    def _keyframe_preview(self, idx, box):
        if self._kf_index is None and self._pv is not None:
            return self._sampler_preview(idx, box)
        try:
            if self._kf_reader is None:
                from AIO_QC.tools.frame_extractor.core.sources import KeyframeReader
                self._kf_reader = KeyframeReader(self._kf_index)
            kidx = self._kf_index.keyframe_before(idx)
            bgr = self._kf_reader.read(kidx)
        except Exception as e:
            logger.warning(f"Keyframe preview failed, using exact seeks: {e}")
            self._kf_index = None
            return None
        if bgr is None:
            return None
        return Frame(kidx, bgr, bgr_to_display(bgr, box, cv2.INTER_LINEAR), preview=True)

    def _sampler_preview(self, idx, box):
        try:
            if self._pv_obj is None:
                from AIO_QC.tools.frame_extractor.core import fast_sampler as FS
                kind, ff = self._pv
                if kind == "pyav":
                    self._pv_obj = FS.PyAVSampler(self.path, self.time_of, self.index_of, keep_full=True,
                                                  full_w=max(640, min(1920, box[0])))
                else:
                    from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffprobe
                    self._pv_obj = FS.FfmpegSampler(ff, self.path, self.width, self.height, self.time_of,
                                                    self.index_of, keep_full=True,
                                                    start_time=FS.probe_start_time(self.path, find_ffprobe(ff)))
            r = self._pv_obj.sample(self.clamp(idx))
        except Exception as e:
            logger.warning(f"Keyframe preview failed, using exact seeks: {e}")
            self._pv = None
            return None
        if r is None or r.full is None:
            return None
        img = np.ascontiguousarray(r.full)
        q = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_BGR888).copy()
        return Frame(r.idx, None, q, preview=True, small=img)

    def _position(self, idx):
        """Make the next read() return frame `idx`. Returns False if aborted
        because a newer seek request arrived."""
        if idx == self._decode_pos:
            return True
        if self._decode_pos < idx <= self._decode_pos + self.seek_grab_limit:
            while self._decode_pos < idx:
                if self._pending is not None:
                    return False
                if not self._grab():
                    break
                self._decode_pos += 1
            if self._decode_pos == idx:
                return True
        self._safe(self.src.seek, idx)   # a failed seek shows up as a failed read -> EOF
        self._decode_pos = idx
        return True

    def run(self):
        consec_drops = 0
        try:
            while True:
                with self._cond:
                    while True:
                        if self._quit:
                            return
                        if (self._pending is not None or self._pending_fast is not None
                                or self._pending_exact is not None):
                            break
                        if self.frame_count > 0 and self._stream_next >= self.frame_count:
                            self._eof = True
                        cap = (self._rt_ahead() if self._pipe_mode else
                               self.max_ahead if self._playing else min(self.max_ahead, self.PAUSED_AHEAD))
                        can_fill = not self._eof and len(self._ahead) < cap
                        now = time.monotonic()
                        if can_fill and (self._playing or now >= self._prefetch_after):
                            break
                        wait = 0.05
                        if can_fill:
                            wait = max(0.001, min(wait, self._prefetch_after - now))
                        self._cond.wait(wait)
                    req, self._pending = self._pending, None
                    fast, self._pending_fast = self._pending_fast, None
                    exact, self._pending_exact = self._pending_exact, None
                    if req is not None:
                        fast = None          # an exact request supersedes previews
                    gen = self._gen
                    idx = self._stream_next
                    pipe_mode = self._pipe_mode and req is None and fast is None
                    late = False
                    if req is None and self._playing and self._clock_t0 is not None and exact is None:
                        now_ = time.monotonic()
                        target = self._clock_target(now_)
                        late = idx < target - (1 if pipe_mode else self.LATE_FRAMES)
                        behind_s = self.time_of(int(target)) - self.time_of(idx) if late else 0.0
                        if behind_s > self.RESYNC_BEHIND_S and now_ - self._last_resync > 0.5:
                            # Decoding is slower than real time (e.g. 8K without GPU decode): skip ahead
                            # like a media player instead of drifting behind the clock.
                            lead = max(self.RESYNC_LEAD_S, 1.3 * self._restart_s) * self._speed
                            jump = self.index_of(self.time_of(int(target)) + lead)
                            if self._keyframes:
                                # land on a keyframe: decoding starts there, no preroll
                                k = bisect.bisect_left(self._keyframes, jump)
                                jump = self._keyframes[k] if k < len(self._keyframes) else idx
                            if self.frame_count:
                                jump = min(jump, self.frame_count - 1)
                            if jump > idx:
                                self._last_resync = now_
                                self.resyncs += 1
                                self.dropped += jump - idx
                                self.drop_resync += jump - idx
                                self._stream_next = idx = jump
                                self._pipe_gen = -1      # restart the pipeline there
                                self._restart_t0 = now_
                                late = False
                    box = self._display_size

                # --- exact frame for a paused pipeline (exact decoder only; pipeline stream untouched)
                if exact is not None:
                    frame = None
                    if self._position(exact):
                        ok, bgr = self._read()
                        if ok:
                            self._decode_pos = exact + 1
                            disp = bgr_to_display(bgr, box, cv2.INTER_AREA) if self.display_images else None
                            frame = Frame(exact, bgr, disp)
                    if frame is not None:
                        with self._cond:
                            self._history.append(frame)
                        self.still_ready.emit(frame)
                    continue

                # --- keyframe preview (does not disturb the main decoder position)
                if fast is not None:
                    frame = self._keyframe_preview(fast, box)
                    with self._cond:
                        deliver = frame is not None and self._pending is None and self._pending_fast is None
                    if deliver:
                        self.still_ready.emit(frame)
                    continue

                # --- seek / positioning request
                if req is not None:
                    idx, emit = req
                    if not self._position(idx) or not emit:
                        continue
                    ok, bgr = self._read()
                    frame = None
                    if ok:
                        self._decode_pos = idx + 1
                        frame = Frame(idx, bgr, bgr_to_display(bgr, box, cv2.INTER_AREA))
                    with self._cond:
                        deliver = False
                        if gen == self._gen:
                            if ok:
                                self._history.append(frame)
                                deliver = self._pending is None
                            else:
                                self._eof = True
                    if deliver:
                        self.still_ready.emit(frame)
                    continue

                # --- real-time pipeline (display-size frames from ffmpeg)
                if pipe_mode:
                    consec_drops = self._fill_from_pipe(idx, gen, late, box, consec_drops)
                    continue

                # --- sequential fill of the read-ahead buffer
                if not self._position(idx):
                    continue
                t_dec = time.perf_counter()
                if late and consec_drops < self.MAX_CONSEC_DROPS:
                    ok = self._grab()      # already late: decode, skip conversion/copy
                    frame = None
                    consec_drops += 1
                else:
                    ok, bgr = self._read()
                    disp = bgr_to_display(bgr, box, cv2.INTER_LINEAR) if (ok and self.display_images) else None
                    frame = Frame(idx, bgr, disp) if ok else None
                    consec_drops = 0
                    self.decode_ms = 0.9 * self.decode_ms + 0.1 * (time.perf_counter() - t_dec) * 1000
                if ok:
                    self._decode_pos = idx + 1
                    self._note_restart()
                with self._cond:
                    if gen != self._gen:
                        continue
                    if not ok:
                        self._eof = True
                        self._found_end(idx)
                        continue
                    self._stream_next = idx + 1
                    if frame is None:
                        self.dropped += 1
                        self.drop_decode += 1
                    else:
                        frame.t_ready = time.monotonic()
                        self._ahead.append(frame)
        except Exception as e:
            logger.exception(f"Decoder thread crashed: {e}")
        finally:
            # Whatever happened, the GUI must see an end of stream (never a
            # player stuck in "playing" with no frames coming).
            with self._cond:
                self._dead = True
                self._eof = True
                self._cond.notify_all()
            self._close_pipe()
            self._safe(self.src.release)
            if self._kf_reader is not None:
                self._safe(self._kf_reader.release)
            if self._pv_obj is not None:
                self._safe(self._pv_obj.close)
