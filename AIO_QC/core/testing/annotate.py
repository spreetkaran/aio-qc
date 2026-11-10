# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Drawing predictions / ground truth on frames, annotated-video export and clip export.

Drawing conventions (same in the app, the reports and exported videos):
  ground truth      dashed, neutral (white with a dark outline so it shows on any background)
  prediction        solid, the model's colour; label "class 0.87" (Inter, see core.lab.textdraw)
  error             solid red with the error type ("wrong class", "duplicate", ...) - only in error views
  missed object     dashed red, "missed"
"""
import os
import subprocess
import threading
import time

import cv2
import numpy as np

from AIO_QC.tools.frame_extractor.core.common import model_hex
from AIO_QC.core.testing.textdraw import put_text, text_size

CRITICAL = (59, 59, 208)          # BGR of #d03b3b
GT_INK = (245, 245, 245)
SHORT = {"cls": "wrong class", "loc": "poor box", "dup": "duplicate", "bkg": "false", "missed": "missed",
         "missed_low_conf": "missed (low conf)"}


def hex_bgr(h):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def model_bgr(slot, mode="dark"):
    return hex_bgr(model_hex(slot, mode))


def _dashed_rect(img, p1, p2, color, t=1, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for a, b, horizontal, fixed in ((x1, x2, True, y1), (x1, x2, True, y2), (y1, y2, False, x1), (y1, y2, False, x2)):
        for s in range(int(a), int(b), dash * 2):
            e = min(int(b), s + dash)
            if horizontal:
                cv2.line(img, (s, int(fixed)), (e, int(fixed)), color, t, cv2.LINE_AA)
            else:
                cv2.line(img, (int(fixed), s), (int(fixed), e), color, t, cv2.LINE_AA)


def _label(img, x, y, text, bg, scale, t=1):
    """A filled label tag whose bottom-left is (x, y), kept inside the image. Text in Inter (core.lab.textdraw);
    scale is the old Hershey scale (0.45 ~ 13 px)."""
    px = max(10, scale * 33)
    tw, th = text_size(text, px)
    padx, pady = max(3, int(px * 0.32)), max(1, int(px * 0.12))
    bw, bh = tw + 2 * padx, th + 2 * pady
    y = int(max(bh, min(int(y), img.shape[0])))
    x = int(max(0, min(x, img.shape[1] - bw)))
    cv2.rectangle(img, (x, y - bh), (x + bw - 1, y - 1), bg, -1)
    lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    put_text(img, text, x + padx, y - bh + pady, px, (20, 20, 20) if lum > 150 else (255, 255, 255))


def draw(img, gt=None, preds=None, names=None, color=(214, 120, 42), conf=0.0, pred_types=None, keep=None,
         gt_types=None, labels=True, scale=None, track_ids=None, gt_names=None, fit_labels=False):
    """Draw on a copy. gt: (boxes, cls); preds: (n, 6) in dataset ids (or model ids with `names`).
    pred_types/keep/gt_types from metrics.classify_errors for an error view.
    fit_labels: thumbnails - leave out a label that would be much bigger than its box (dense scenes would
    otherwise be buried under text); the box keeps its colour. Full-size plots draw every label."""
    out = img.copy()
    h, w = out.shape[:2]
    s = scale or max(0.4, min(1.4, w / 1600))
    t = max(1, int(round((1.3 if fit_labels else 2) * s)))       # thumbnails: lighter lines
    names = names or {}
    gt_names = gt_names or names
    lab_h = text_size("Ag", max(10, 14.85 * s))[1] + 4

    def fits(b, txt):
        if not fit_labels:
            return True
        bw, bh = b[2] - b[0], b[3] - b[1]
        return bh >= lab_h * 1.3 and bw >= text_size(txt, max(10, 14.85 * s))[0] * 0.85
    if gt is not None and gt[0] is not None:
        for k, (b, c) in enumerate(zip(np.asarray(gt[0]).reshape(-1, 4), np.asarray(gt[1]).reshape(-1))):
            missed = gt_types is not None and gt_types[k] != "tp"
            p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            if not missed:
                _dashed_rect(out, p1, p2, (30, 30, 30), t + 2, int(10 * s))
            _dashed_rect(out, p1, p2, CRITICAL if missed else GT_INK, t, int(10 * s))
            txt = f"{gt_names.get(int(c), c)} · {SHORT[gt_types[k]]}" if missed else ""
            if labels and missed and fits(b, txt):
                _label(out, p1[0], p2[1] + 1 + lab_h, txt, CRITICAL, 0.45 * s)
    if preds is not None and len(preds):
        types = {}
        if pred_types is not None and keep is not None:
            types = {int(k): ty for k, ty in zip(keep, pred_types)}
        order = np.argsort(preds[:, 4])                      # most confident drawn last (on top)
        for j in order:
            b = preds[j]
            if b[4] < conf and j not in types:
                continue
            if types and j not in types:
                continue
            ty = types.get(int(j), "tp")
            col = color if ty == "tp" else CRITICAL
            p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            cv2.rectangle(out, p1, p2, col, t, cv2.LINE_AA)
            if labels:
                base = f"{names.get(int(b[5]), int(b[5]))} {b[4]:.2f}"
                if track_ids is not None and j < len(track_ids) and track_ids[j] >= 0:
                    base = f"#{track_ids[j]} " + base
                txt = base + (f" · {SHORT[ty]}" if ty != "tp" else "")
                if not fits(b, txt):
                    txt = base if fits(b, base) else ""          # thumbnails: short form, or colour only
            if labels and txt:
                _label(out, p1[0] - t // 2, p1[1] - t // 2, txt, col, 0.45 * s)
    return out


def thumb(img, max_w=360, quality=80):
    """JPEG bytes of a downscaled frame."""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def focus_rect(shape, boxes, pad=0.25, min_size=320):
    """(x0, y0, w, h) of a frame-aspect crop around the boxes."""
    h, w = shape[:2]
    if boxes is None or not len(boxes):
        return 0, 0, w, h
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cw, ch = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    cw = min(w, max(cw, min_size, ch * w / h))
    ch = min(h, cw * h / w)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, c = int(max(0, min(w - cw, cx - cw / 2))), int(max(0, min(h - ch, cy - ch / 2)))
    return a, c, int(cw), int(ch)


def crop_focus(img, boxes, pad=0.25, min_size=320):
    """Crop around the boxes of interest (so small errors are visible in a thumbnail). Returns (crop, (x0, y0))."""
    a, c, cw, ch = focus_rect(img.shape, boxes, pad, min_size)
    return img[c:c + ch, a:a + cw], (a, c)


# ================================================================== encoders
_ENC_CACHE = {}
ENCODERS = [   # (name, args for "high" quality) - tried in this order; the first that works here wins
    ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
    ("hevc_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-tag:v", "hvc1"]),
    ("h264_qsv", ["-global_quality", "20", "-look_ahead", "0"]),
    ("h264_amf", ["-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"]),
    ("h264_videotoolbox", ["-q:v", "65"]),
    ("libx264", ["-preset", "veryfast", "-crf", "18"]),
    ("libopenh264", ["-b:v", "20M"]),
    ("mpeg4", ["-q:v", "3"]),
]
ENCODER_LABELS = {"h264_nvenc": "NVIDIA NVENC (H.264)", "hevc_nvenc": "NVIDIA NVENC (HEVC)", "h264_qsv": "Intel Quick Sync",
                  "h264_amf": "AMD AMF", "h264_videotoolbox": "Apple VideoToolbox", "libx264": "x264 (CPU)",
                  "libopenh264": "OpenH264 (CPU)", "mpeg4": "MPEG-4 (CPU)"}


def available_encoders(ffmpeg):
    try:
        from AIO_QC.core import procs
        out = procs.run([ffmpeg, "-hide_banner", "-encoders"], timeout=15, text=True).stdout
    except Exception:
        return []
    names = {l.split()[1] for l in out.splitlines() if len(l.split()) > 1 and l.strip()[:1] == "V"}
    return [e for e, _ in ENCODERS if e in names]


def pick_encoder(ffmpeg, w, h, prefer="auto"):
    """First encoder that really works on this machine for this size (a 4-frame trial encode, cached)."""
    from AIO_QC.core import procs
    key = (ffmpeg, w > 4096 or h > 4096, prefer)
    if key in _ENC_CACHE:
        return _ENC_CACHE[key]
    cands = available_encoders(ffmpeg)
    if prefer != "auto" and prefer in cands:
        cands = [prefer] + [c for c in cands if c != prefer]
    tw, th = (min(w, 1280) // 2 * 2, min(h, 720) // 2 * 2)
    if w > 4096 or h > 4096:            # H.264 hardware encoders stop at 4096: test the real size
        tw, th = w // 2 * 2, h // 2 * 2
    for name in cands:
        args = dict(ENCODERS)[name]
        try:
            r = procs.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                           f"color=c=gray:s={tw}x{th}:r=25:d=0.16", "-c:v", name, *args, "-pix_fmt", "yuv420p",
                           "-f", "null", "-"], timeout=30, text=True)
            if r.returncode == 0:
                _ENC_CACHE[key] = name
                return name
        except Exception:
            continue
    _ENC_CACHE[key] = None
    return None


# ================================================================== tracking IDs
class IdTracker:
    """Persistent object IDs for drawing (greedy IoU, class-aware, survives short gaps)."""

    def __init__(self, iou=0.3, max_age=15):
        self.iou, self.max_age = iou, max_age
        self.tracks = []      # [id, box, cls, age]
        self.next = 1

    def update(self, dets):
        from AIO_QC.core.testing.metrics import box_iou
        ids = -np.ones(len(dets), int)
        if len(dets) and self.tracks:
            tb = np.array([t[1] for t in self.tracks], np.float32)
            iou = box_iou(tb, dets[:, :4]) * (np.array([t[2] for t in self.tracks])[:, None] == dets[:, 5][None, :])
            used_t = set()
            for ti, di in sorted(((ti, di) for ti in range(len(tb)) for di in range(len(dets))),
                                 key=lambda p: -iou[p]):
                if iou[ti, di] < self.iou:
                    break
                if ti in used_t or ids[di] >= 0:
                    continue
                used_t.add(ti)
                ids[di] = self.tracks[ti][0]
                self.tracks[ti][1], self.tracks[ti][3] = dets[di, :4].tolist(), 0
        for di in np.nonzero(ids < 0)[0]:
            ids[di] = self.next
            self.tracks.append([self.next, dets[di, :4].tolist(), dets[di, 5], 0])
            self.next += 1
        for t in self.tracks:
            t[3] += 1
        self.tracks = [t for t in self.tracks if t[3] <= self.max_age]
        return ids


# ================================================================== annotated video
def _hud(img, lines, s):
    pad = int(10 * s)
    y = pad
    px = 16 * s
    for txt, col in lines:
        tw, th = text_size(txt, px, "Medium")
        tx = pad + int(18 * s) if col is not None else pad
        cv2.rectangle(img, (pad - int(6 * s), y - int(3 * s)), (tx + tw + int(6 * s), y + th + int(3 * s)), (20, 20, 20), -1)
        if col is not None:
            cv2.line(img, (pad, y + th // 2), (pad + int(12 * s), y + th // 2), col, max(2, int(3 * s)), cv2.LINE_AA)
        put_text(img, txt, tx, y, px, (240, 240, 240), "Medium")
        y += th + int(10 * s)


def export_annotated(video, out_path, specs, cache, ffmpeg, start=None, end=None, layout="overlay", conf=0.25,
                     slots=None, names_override=None, tracks=True, hud=True, encoder="auto", audio=True,
                     max_width=3840, progress=None, cancel=None, batch=8, log=None):
    """Draw 1-4 models on a video and encode it. Cached predictions are used when present (from a model run or
    an earlier export); missing frames are inferred in batches and cached. layout: 'overlay' (all models on
    one picture) | 'side_by_side' (one panel per model, 2 = left/right, 3-4 = 2x2).
    Returns dict(path, frames, encoder, seconds) - the output file is written to <out>.part and renamed at the
    end, so a cancelled / failed export never leaves a half file under the real name."""
    from AIO_QC.core import procs
    from AIO_QC.core.testing.runner import video_key, video_frames, frame_range
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    progress = progress or (lambda *a: None)
    cancel = cancel or (lambda: False)
    log = log or (lambda m: None)
    t0 = time.time()
    slots = slots or list(range(len(specs)))
    confs = conf if isinstance(conf, (list, tuple)) else [conf] * len(specs)
    src = open_source(video, "auto", hw_accel=False)
    fps, W, H = src.fps, src.width, src.height
    a, b = frame_range(src, start, end)
    t_start = src.time_of(a)
    src.release()
    n_frames = b - a + 1
    vk = video_key(video)
    names = []
    dets = {}
    for i, s in enumerate(specs):
        m = cache.model(s.key) or {}
        names.append((names_override or {}).get(i) or m.get("names") or {})
    # output geometry
    n = len(specs)
    if layout == "side_by_side" and n > 1:
        cols, rows = (2, 1) if n == 2 else (2, 2)
        pw = min(W, max_width // cols) // 2 * 2
        ph = int(round(H * pw / W)) // 2 * 2
        OW, OH = pw * cols, ph * rows
    else:
        cols = rows = 1
        sc = min(1.0, max_width / W)
        OW, OH = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
        pw, ph = OW, OH
    enc = pick_encoder(ffmpeg, OW, OH, encoder)
    if enc is None:
        raise RuntimeError("No working video encoder found in ffmpeg")
    part = out_path + ".part.mp4"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{OW}x{OH}", "-r", f"{fps:.6f}", "-i", "-"]
    if audio:
        cmd += ["-ss", f"{t_start:.6f}", "-i", video, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k",
                "-shortest"]
    cmd += ["-c:v", enc, *dict(ENCODERS)[enc], "-pix_fmt", "yuv420p", "-movflags", "+faststart", part]
    proc = procs.popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err = []
    threading.Thread(target=lambda: err.extend(proc.stderr.read().decode("utf-8", "replace").splitlines()[-5:]),
                     daemon=True).start()
    trackers = [IdTracker() for _ in specs]
    cached = [cache.get_many(s.key, [f"{vk}#{i}" for i in range(a, b + 1)]) for s in specs]
    missing = [len(cached[i]) < n_frames for i in range(n)]
    for i, s in enumerate(specs):
        if missing[i]:
            progress(0, n_frames, f"loading {s.label}…")
            dets[i] = s.load()
            if not names[i]:
                names[i] = dict(getattr(dets[i], "names", None) or {})
            cache.set_model(s.key, s, names[i], dict(backend=getattr(dets[i], "provider", "")))
    s_ = max(0.6, pw / 1280)
    written = 0
    ok = False
    try:
        buf = []

        def flush(buf):
            nonlocal written
            frames = [f for _, f in buf]
            preds = []
            for i, s in enumerate(specs):
                keys = [f"{vk}#{k}" for k, _ in buf]
                have = [cached[i].get(k) for k in keys]
                todo = [j for j, h in enumerate(have) if h is None]
                got = {}
                if todo:
                    from AIO_QC.core.testing.runner import _infer
                    res = _infer(dets[i], [frames[j] for j in todo], s.conf)
                    got = {j: r for j, r in zip(todo, res)}
                    cache.put_many(s.key, [(keys[j], got[j], 0.0) for j in todo])
                preds.append([have[j][0] if have[j] is not None else got[j] for j in range(len(buf))])
            for j, (k, frame) in enumerate(buf):
                panels = []
                for i in range(n):
                    p = preds[i][j]
                    p = p[p[:, 4] >= confs[i]] if len(p) else p
                    ids = trackers[i].update(p) if tracks and len(p) else None
                    if layout == "side_by_side" and n > 1:
                        im = draw(frame, None, p, names[i], model_bgr(slots[i]), 0.0, track_ids=ids, scale=max(0.6, W / 1280))
                        im = cv2.resize(im, (pw, ph), interpolation=cv2.INTER_AREA)
                        if hud:
                            _hud(im, [(specs[i].label, model_bgr(slots[i]))], s_)
                        panels.append(im)
                    else:
                        frame = draw(frame, None, p, names[i], model_bgr(slots[i]), 0.0, track_ids=ids,
                                     scale=max(0.6, W / 1280))
                if panels:
                    while len(panels) < cols * rows:
                        panels.append(np.zeros_like(panels[0]))
                    out = np.vstack([np.hstack(panels[r * cols:(r + 1) * cols]) for r in range(rows)])
                else:
                    out = frame if (OW, OH) == (W, H) else cv2.resize(frame, (OW, OH), interpolation=cv2.INTER_AREA)
                    if hud:
                        _hud(out, [(s.label, model_bgr(slots[i])) for i, s in enumerate(specs)], s_)
                if hud:
                    from AIO_QC.tools.frame_extractor.core.sources import format_time
                    t = format_time(t_start + (k - a) / fps)
                    txt = f"frame {k}  {t}"
                    tw, th = text_size(txt, 16 * s_, "Medium")
                    cv2.rectangle(out, (OW - tw - int(24 * s_), OH - th - int(20 * s_)), (OW, OH), (20, 20, 20), -1)
                    put_text(out, txt, OW - tw - int(14 * s_), OH - th - int(10 * s_), 16 * s_, (240, 240, 240), "Medium")
                proc.stdin.write(np.ascontiguousarray(out).tobytes())
                written += 1
        for k, t, frame in video_frames(video, 1, start, end, cancel):
            buf.append((k, frame))
            if len(buf) >= batch:
                flush(buf)
                buf = []
                progress(written, n_frames, f"frame {k}")
            if cancel():
                break
        if buf and not cancel():
            flush(buf)
        proc.stdin.close()
        rc = proc.wait()
        if cancel():
            raise InterruptedError("cancelled")
        if rc != 0:
            raise RuntimeError("ffmpeg: " + " | ".join(err[-3:]))
        os.replace(part, out_path)
        ok = True
        progress(n_frames, n_frames, "done")
        return dict(path=out_path, frames=written, encoder=enc, seconds=round(time.time() - t0, 1),
                    size=(OW, OH))
    finally:
        if not ok:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                os.remove(part)
            except OSError:
                pass


# ================================================================== clips
_KF = {}


def keyframe_before(video, t):
    """Time of the keyframe at or before t (from the MP4 / MOV index; cached), or None when unknown."""
    if video not in _KF:
        try:
            from AIO_QC.tools.frame_extractor.core.mp4index import mp4_index
            idx = mp4_index(video)
            _KF[video] = sorted(idx[0][k] for k in idx[1]) if idx else None
        except Exception:
            _KF[video] = None
    kf = _KF[video]
    if not kf:
        return None
    import bisect
    i = bisect.bisect_right(kf, t + 1e-6) - 1
    return kf[i] if i >= 0 else 0.0


def export_clip(video, out_path, start, end, ffmpeg, mode="fast", encoder="auto", audio=True, progress=None,
                cancel=None):
    """Cut [start, end] seconds.
      fast  - stream copy: lossless and instant, but the clip begins at the keyframe at or before `start`
      exact - re-encode: starts exactly at `start` (frame-accurate), encoder auto-picked (GPU when available)
    Returns dict(path, mode, start, end, encoder)."""
    from AIO_QC.core import procs
    progress = progress or (lambda *a: None)
    cancel = cancel or (lambda: False)
    start, end = max(0.0, float(start)), float(end)
    if end <= start:
        raise ValueError("The end must be after the start")
    real_start = start
    if mode == "fast":
        from AIO_QC.tools.frame_extractor.core.mp4index import mp4_index
        idx = mp4_index(video)
        if idx:
            ts, keys = idx
            prior = [ts[k] for k in keys if ts[k] <= start + 1e-6]
            real_start = prior[-1] if prior else 0.0
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.6f}", "-to", f"{end:.6f}",
               "-i", video, "-map", "0:v:0"] + (["-map", "0:a?"] if audio else []) + \
              ["-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart"]
        enc = "copy"
    else:
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        s = open_source(video, "auto", hw_accel=False)
        w, h = s.width, s.height
        s.release()
        enc = pick_encoder(ffmpeg, w, h, encoder) or "libx264"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.6f}", "-i", video,
               "-t", f"{end - start:.6f}", "-map", "0:v:0"] + (["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"] if audio else []) + \
              ["-c:v", enc, *dict(ENCODERS).get(enc, []), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    part = out_path + ".part" + os.path.splitext(out_path)[1]
    cmd += ["-progress", "pipe:1", "-nostats", part]
    proc = procs.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    err = []
    threading.Thread(target=lambda: err.extend(proc.stderr.read().decode("utf-8", "replace").splitlines()[-5:]),
                     daemon=True).start()
    dur = end - start
    ok = False
    try:
        for line in proc.stdout:
            if cancel():
                proc.kill()
                raise InterruptedError("cancelled")
            line = line.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    progress(max(0.0, min(1.0, int(line.split("=")[1]) / 1e6 / max(dur, 1e-6))), 1.0, "")
                except ValueError:
                    pass
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("ffmpeg: " + " | ".join(err[-3:]))
        os.replace(part, out_path)
        ok = True
        return dict(path=out_path, mode=mode, start=real_start, end=end, encoder=enc)
    finally:
        if not ok:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                os.remove(part)
            except OSError:
                pass
