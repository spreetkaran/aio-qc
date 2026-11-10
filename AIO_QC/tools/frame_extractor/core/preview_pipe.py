# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Real-time playback pipeline (Qt-free).

Why: at 8K every decoded frame is ~95 MB of BGR. The exact-frame decoder (OpenCV / PyAV)
converts each full-resolution frame on the CPU and only then shrinks it for the screen, which
caps 8K at ~10-20 fps even when the GPU decodes. For *playback* only a screen-sized picture is
needed, so ffmpeg decodes AND scales on the GPU (or with threaded swscale) and pipes frames that
are already display size - the CPU copies ~6 MB instead of ~95 MB per frame.

Pausing, stepping, saving, zooming and analysis still use the exact full-resolution decoder.

Chains (first that works on this machine wins; probed once per ffmpeg / codec / size, cached):
    cuda          NVDEC decode + scale_cuda          NVIDIA
    qsv           Quick Sync decode + scale_qsv      Intel
    vaapi         VAAPI decode + scale_vaapi         Linux Intel / AMD
    videotoolbox  VT decode + scale_vt               macOS
    d3d11va       D3D11 decode + threaded swscale    Windows (any GPU)
    auto          -hwaccel auto + threaded swscale
    cpu           software decode + threaded swscale
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from AIO_QC.core import procs



CHAINS = {
    "cuda": (["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
             "scale_cuda=w={w}:h={h}:format=nv12,hwdownload,format=nv12"),
    "qsv": (["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"],
            "scale_qsv=w={w}:h={h}:format=nv12,hwdownload,format=nv12"),
    "vaapi": (["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"],
              "scale_vaapi=w={w}:h={h}:format=nv12,hwdownload,format=nv12"),
    "videotoolbox": (["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"],
                     "scale_vt=w={w}:h={h},hwdownload,format=nv12"),
    "d3d11va": (["-hwaccel", "d3d11va"], "scale={w}:{h}:flags=fast_bilinear"),
    "auto": (["-hwaccel", "auto"], "scale={w}:{h}:flags=fast_bilinear"),
    "cpu": ([], "scale={w}:{h}:flags=fast_bilinear"),
}
CHAIN_LABELS = {"cuda": "NVIDIA CUDA", "qsv": "Intel Quick Sync", "vaapi": "VAAPI", "videotoolbox": "VideoToolbox",
                "d3d11va": "Direct3D 11", "auto": "GPU (auto)", "cpu": "CPU"}


def _hw_env():
    env = dict(os.environ)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")     # ffmpeg's cuda:N == nvidia-smi index N
    return env


def find_ffmpeg(user_path=""):
    """User setting > PATH > imageio-ffmpeg's bundled binary."""
    if user_path and os.path.isfile(user_path):
        return user_path
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_info(ffmpeg):
    """(version tuple, set of hwaccels, set of filters)."""
    def run(*a):
        try:
            return procs.run([ffmpeg, "-hide_banner", *a], text=True, timeout=15).stdout
        except Exception:
            return ""
    m = re.search(r"version\s+n?(\d+)\.(\d+)", run("-version"))
    ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    hw = {l.strip() for l in run("-hwaccels").splitlines()[1:] if l.strip()}
    filters = {l.split()[1] for l in run("-filters").splitlines() if len(l.split()) > 2 and l.split()[1].startswith("scale")}
    return ver, hw, filters


def candidate_chains(ffmpeg):
    ver, hw, filters = ffmpeg_info(ffmpeg)
    out = []
    for name, (_, vf) in CHAINS.items():
        needs_hw = name not in ("auto", "cpu")
        needs_filter = vf.split("=")[0]
        if needs_hw and name not in hw:
            continue
        if needs_filter != "scale" and needs_filter not in filters:
            continue
        if name == "d3d11va" and sys.platform != "win32":
            continue
        if name == "videotoolbox" and sys.platform != "darwin":
            continue
        out.append(name)
    return out, ver


def display_size(src_w, src_h, box_w, box_h):
    """Fit into the box, never upscale, even dimensions (NV12)."""
    s = min(box_w / src_w, box_h / src_h, 1.0) if box_w > 0 and box_h > 0 else 1.0
    return max(2, int(src_w * s) // 2 * 2), max(2, int(src_h * s) // 2 * 2)


class PreviewPipe:
    """One ffmpeg process decoding from `start_sec`, emitting display-size BGR frames in order."""

    def __init__(self, ffmpeg, path, chain, w, h, start_sec=0.0, threads=0, version=(6, 0), pix_fmt="bgr24",
                 frames=0, background=False):
        """pix_fmt 'bgr24': display-size BGR (software canvas); 'nv12': planes for the GPU canvas
        (w, h = full resolution -> original quality, YUV->RGB happens in the shader)."""
        self.w, self.h = w, h
        self.pix_fmt = pix_fmt
        self.frame_bytes = w * h * 3 if pix_fmt == "bgr24" else w * h * 3 // 2
        in_args, vf = CHAINS[chain]
        sync = ["-fps_mode", "passthrough"] if version >= (5, 1) else ["-vsync", "passthrough"]
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *in_args]
        if threads:
            cmd += ["-threads", str(threads)]
        if start_sec > 0:
            cmd += ["-ss", f"{start_sec:.6f}"]             # input seek (keyframe) + exact trim to the frame
        cmd += ["-i", path, "-an", "-sn", "-dn"] + (["-frames:v", str(int(frames))] if frames else []) + [
                "-vf", vf.format(w=w, h=h) + f",format={pix_fmt}",
                "-filter_threads", str(max(1, os.cpu_count() or 1)), *sync,
                "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
        self.cmd = cmd
        self.proc = procs.popen(cmd, background=background, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, bufsize=self.frame_bytes * 2, env=_hw_env())
        self._err = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.proc.stderr:
            if len(self._err) < 20:
                self._err.append(line.decode("utf-8", "replace").rstrip())

    def read(self):
        """Next frame (h, w, 3) uint8, or None at the end / on error."""
        buf = bytearray(self.frame_bytes)
        view, got = memoryview(buf), 0
        try:
            while got < self.frame_bytes:
                n = self.proc.stdout.readinto(view[got:])
                if not n:
                    return None
                got += n
        except (OSError, ValueError):
            return None
        a = np.frombuffer(buf, np.uint8)
        if self.pix_fmt == "nv12":
            n = self.w * self.h
            return a[:n].reshape(self.h, self.w), a[n:].reshape(self.h // 2, self.w // 2, 2)
        return a.reshape(self.h, self.w, 3)

    @property
    def error(self):
        return " | ".join(self._err[-3:])

    def close(self):
        """Kill without blocking the caller (UI): reaping happens on a background thread."""
        proc = self.proc
        try:
            proc.kill()
        except Exception:
            pass

        def reap():
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        threading.Thread(target=reap, daemon=True).start()


def benchmark(ffmpeg, path, chain, w, h, frames=45, timeout=20, version=(6, 0), pix_fmt="bgr24"):
    """Decode (+scale) speed of one chain (fps), or None if it doesn't work here."""
    pipe = PreviewPipe(ffmpeg, path, chain, w, h, 0.0, version=version, pix_fmt=pix_fmt)
    t0, n = time.perf_counter(), 0
    try:
        while n < frames and time.perf_counter() - t0 < timeout:
            if pipe.read() is None:
                break
            n += 1
    finally:
        pipe.close()
    dt = time.perf_counter() - t0
    return (n / dt) if n >= min(frames, 5) else None


def benchmark_parallel(ffmpeg, path, chain, w, h, keyframes, time_of, frame_count, workers, version=(6, 0),
                       pix_fmt="nv12", frames=120, timeout=25):
    """Throughput (fps) of segment-parallel decoding from the start of the video, or None."""
    pp = ParallelPipe(ffmpeg, path, chain, w, h, 0, keyframes, time_of, frame_count, workers, version, pix_fmt)
    t0, n = time.perf_counter(), 0
    try:
        while n < frames and time.perf_counter() - t0 < timeout:
            if pp.read() is None:
                break
            n += 1
    finally:
        pp.close()
    dt = time.perf_counter() - t0
    return (n / dt, pp.workers) if n >= min(frames, 10) else (None, 0)


def choose_chain(ffmpeg, path, w, h, need_fps, progress=None):
    """Try the candidate chains in order of preference; the first fast enough wins, else the fastest.
    Returns (chain, fps, results)."""
    cands, ver = candidate_chains(ffmpeg)
    results = {}
    for c in cands:
        if progress:
            progress(c)
        fps = benchmark(ffmpeg, path, c, w, h, version=ver)
        results[c] = fps
        if fps and fps >= need_fps * 1.15:
            return c, fps, results
    ok = {c: f for c, f in results.items() if f}
    if not ok:
        return None, 0.0, results
    best = max(ok, key=ok.get)
    return best, ok[best], results


def find_ffprobe(ffmpeg):
    """ffprobe next to ffmpeg, or on PATH (imageio-ffmpeg ships no ffprobe)."""
    if ffmpeg:
        d, name = os.path.split(ffmpeg)
        cand = os.path.join(d, name.replace("ffmpeg", "ffprobe"))
        if cand != ffmpeg and os.path.isfile(cand):
            return cand
    return shutil.which("ffprobe")


def keyframes_ffprobe(path, ffprobe, timeout=120):
    """Keyframe frame indices by demuxing packets (no decoding): fast even for long 8K files.
    Returns (keyframe_indices, frame_count) or None."""
    try:
        out = procs.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "packet=pts_time,dts_time,flags", "-of", "csv=p=0", path],
                             text=True, timeout=timeout).stdout
    except Exception:
        return None
    pk = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        t = parts[0] if parts[0] not in ("", "N/A") else parts[1]
        try:
            pk.append((float(t), "K" in parts[2]))
        except ValueError:
            continue
    if not pk:
        return None
    pk.sort(key=lambda x: x[0])
    return [i for i, (_, k) in enumerate(pk) if k], len(pk)


def probe_color(path, ffprobe):
    """(color_space, color_range) tags of the video stream (header only)."""
    if not ffprobe:
        return "", ""
    try:
        # key=value output: ffprobe's CSV lists fields in its own order (color_range before color_space)
        out = procs.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=color_space,color_range", "-of", "default=noprint_wrappers=1", path],
                             text=True, timeout=10).stdout
        kv = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        return kv.get("color_space", ""), kv.get("color_range", "")
    except Exception:
        return "", ""


class ParallelPipe:
    """Segment-parallel decoding: N ffmpeg processes decode consecutive keyframe-aligned segments at the same
    time (each GOP decodes independently); frames come out strictly in order. Throughput scales with the
    processes when one decoder can't keep up (CPU decoding, or 8K H.264 that GPU decoders don't support).
    Memory: segments decoded ahead are held until shown - `workers` is capped by `budget` bytes.
    Same interface as PreviewPipe (read / close / error / pix_fmt)."""

    def __init__(self, ffmpeg, path, chain, w, h, start_idx, keyframes, time_of, frame_count, workers=4,
                 version=(6, 0), pix_fmt="nv12", budget=None, min_seconds=1.0, fps=30.0):
        self.args = (ffmpeg, path, chain, w, h, version, pix_fmt)
        self.pix_fmt, self.w, self.h = pix_fmt, w, h
        self.time_of, self.frame_count = time_of, frame_count
        fb = w * h * 3 // (2 if pix_fmt == "nv12" else 1)
        if budget is None:
            from AIO_QC.core.procs import decode_ahead_budget
            budget = decode_ahead_budget()
        # big frames: keep segments to one GOP so each decoder buffers as little as possible
        if fb > 20 * 2 ** 20:
            min_seconds = 0.0
        # segments: [start, next keyframe) then whole GOPs merged to >= min_seconds
        keys = [k for k in keyframes if k > start_idx]
        bounds, cur = [start_idx], start_idx
        for k in keys:
            if time_of(k) - time_of(bounds[-1]) >= min_seconds or bounds[-1] == start_idx and k > start_idx:
                bounds.append(k)
        bounds.append(frame_count)
        self.segments = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
        seg_frames = max((b - a for a, b in self.segments[:8]), default=1)
        self.workers = int(max(1, min(workers, budget // max(1, fb * seg_frames))))
        self._next_seg = 0               # next segment to launch
        self._cur = None                 # segment being read: dict
        self._live = []                  # launched, unread segments (in order)
        self._err = []
        self._closed = False
        self._lock = threading.Lock()
        for _ in range(self.workers):
            self._launch()

    def _launch(self):
        if self._next_seg >= len(self.segments) or self._closed:
            return
        a, b = self.segments[self._next_seg]
        self._next_seg += 1
        ffmpeg, path, chain, w, h, version, pix_fmt = self.args
        t = self.time_of(a)
        if a > 0:
            t -= max(1e-4, (t - self.time_of(a - 1)) / 4)
        seg = dict(a=a, b=b, frames=[], done=False, cond=threading.Condition())
        try:
            seg["pipe"] = PreviewPipe(ffmpeg, path, chain, w, h, max(0.0, t), version=version, pix_fmt=pix_fmt,
                                      threads=2, frames=b - a)
        except Exception as e:
            self._err.append(str(e))
            seg["done"] = True
            self._live.append(seg)
            return

        def reader():
            n = 0
            while n < b - a and not self._closed:
                fr = seg["pipe"].read()
                if fr is None:
                    break
                with seg["cond"]:
                    seg["frames"].append(fr)
                    seg["cond"].notify_all()
                n += 1
            if n < b - a and not self._closed and seg["pipe"].error:
                self._err.append(seg["pipe"].error)
            with seg["cond"]:
                seg["done"] = True
                seg["cond"].notify_all()
        threading.Thread(target=reader, daemon=True).start()
        self._live.append(seg)

    def read(self):
        while not self._closed:
            if self._cur is None:
                if not self._live:
                    return None
                self._cur = self._live.pop(0)
                self._cur["i"] = 0
            seg = self._cur
            with seg["cond"]:
                while seg["i"] >= len(seg["frames"]) and not seg["done"] and not self._closed:
                    seg["cond"].wait(0.5)
                if seg["i"] < len(seg["frames"]):
                    fr = seg["frames"][seg["i"]]
                    seg["frames"][seg["i"]] = None      # release memory as soon as it's handed over
                    seg["i"] += 1
                    return fr
            # segment finished: it must have produced all its frames, otherwise the stream is broken
            short = seg["i"] < seg["b"] - seg["a"]
            seg["pipe"].close() if "pipe" in seg else None
            self._cur = None
            self._launch()                               # keep `workers` segments in flight
            if short:
                if not self._err:
                    self._err.append(f"segment {seg['a']}-{seg['b']} ended after {seg['i']} frames")
                return None
        return None

    @property
    def error(self):
        return " | ".join(self._err[-3:])

    def close(self):
        self._closed = True
        for seg in ([self._cur] if self._cur else []) + self._live:
            if seg and "pipe" in seg:
                seg["pipe"].close()
            if seg:
                with seg["cond"]:
                    seg["cond"].notify_all()
