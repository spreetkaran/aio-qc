# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Extraction markers: frames the user picks on the timeline (Qt-free).

  suggest()            propose marker positions from the timeline analysis
                       (scene starts, sharpest per window, model-flagged, uniform)
  snap_to_sharpest()   move a marker to the sharpest frame within +-radius frames
                       (marks placed while playing often land on motion blur)
  save_marked_frame()  write one frame (+ LabelMe / YOLO pre-annotations, dataset hash)
  MarkerExtractor      owns a decoder; used by the GUI worker (auto-extract / extract pending)
"""
import os


from AIO_QC.tools.frame_extractor.core.metrics import sharpness
from AIO_QC.tools.frame_extractor.core.sources import open_source

SUGGEST_MODES = {
    "scenes": "Start of every scene (first sharp sample after each cut)",
    "sharpest": "Sharpest sample in every window of N seconds",
    "flagged": "Samples the models flagged (uncertain / disagreement)",
    "uniform": "Every N seconds",
}


def suggest(samples, mode, lo=0, hi=None, fps=30.0, every_s=2.0, min_sharp=0.0, cuts=(), timestamps=None):
    """samples: {frame: (sharpness, motion, flagged)} from the timeline analysis. Returns sorted frames."""
    keys = sorted(k for k in samples if k >= lo and (hi is None or k <= hi))
    if not keys:
        return []
    t = (lambda f: timestamps[f]) if timestamps else (lambda f: f / fps)
    ok = lambda f: samples[f][0] >= min_sharp
    out = []
    if mode == "flagged":
        out = [k for k in keys if samples[k][2]]
    elif mode == "scenes":
        starts = [keys[0]] + sorted(c for c in cuts if lo <= c <= (hi if hi is not None else keys[-1]))
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else keys[-1] + 1
            seg = [k for k in keys if s <= k < end]
            sharp = [k for k in seg if ok(k)]
            if seg:
                out.append((sharp or seg)[0] if not sharp else max(sharp[:max(1, len(sharp) // 3)],
                                                                    key=lambda k: samples[k][0]))
    else:   # windows of every_s seconds
        win, cur = [], None
        for k in keys:
            w = int(t(k) // max(1e-6, every_s))
            if cur is not None and w != cur:
                out.append(_pick(win, samples, mode, ok))
                win = []
            cur = w
            win.append(k)
        if win:
            out.append(_pick(win, samples, mode, ok))
        out = [k for k in out if k is not None]
    return sorted(set(out))


def _pick(win, samples, mode, ok):
    if mode == "uniform":
        return win[0]
    cands = [k for k in win if ok(k)]
    return max(cands, key=lambda k: samples[k][0]) if cands else None


def snap_to_sharpest(src, idx, radius, roi=None, pos=None):
    """Decode idx-radius .. idx+radius sequentially; return (best_frame, best_sharpness, bgr)."""
    lo = max(0, idx - radius)
    hi = min(src.frame_count - 1, idx + radius) if src.frame_count else idx + radius
    src.seek(lo)
    best = (idx, -1.0, None)
    for f in range(lo, hi + 1):
        ok, img = src.read()
        if not ok:
            break
        crop = img[roi[1]:roi[3], roi[0]:roi[2]] if roi else img
        s = sharpness(crop)
        # prefer the original frame on ties, then the closest one
        if s > best[1] + 1e-9 or (abs(s - best[1]) <= 1e-9 and abs(f - idx) < abs(best[0] - idx)):
            best = (f, s, img)
    return best


def save_marked_frame(img, idx, name, out_dir, ext=".jpg", detectors=None, preannot="none", det_conf=0.25,
                      hdb=None, video_key=None, quality=95, namer=None, roi=None, kind="marker"):
    """Write one picked frame (already cropped) and its pre-annotations.
    Returns the path relative to out_dir, or None when the naming policy skipped it."""
    from AIO_QC.tools.frame_extractor.core.extract import _imwrite
    from AIO_QC.tools.frame_extractor.core.naming import Namer
    namer = namer or Namer(video_path=video_key or name)
    path, _ = namer.path(out_dir, idx, ext, img, kind, "", roi)
    if path is None:
        return None
    _imwrite(path, img, ext, quality)
    if video_key and os.path.isfile(video_key):
        from AIO_QC.tools.frame_extractor.core import frame_sources
        t = namer.time_of(idx) if getattr(namer, "time_of", None) else None
        frame_sources.record(path, video_key, idx, t)
    fname = os.path.relpath(path, out_dir).replace(os.sep, "/")
    if detectors and preannot != "none":
        dets = [d for d in detectors[0].detect(img) if d.conf >= det_conf]
        h, w = img.shape[:2]
        stem = os.path.splitext(path)[0]
        if preannot == "labelme":
            from AIO_QC.tools.frame_extractor.core.detector import write_labelme
            write_labelme(stem + ".json", os.path.basename(path), w, h, dets, flags={"manual": True})
        elif preannot == "yolo":
            from AIO_QC.tools.frame_extractor.core.detector import write_yolo
            write_yolo(stem + ".txt", w, h, dets)
            names = getattr(detectors[0], "names", None)
            cls_file = os.path.join(os.path.dirname(path), "classes.txt")
            if names and not os.path.exists(cls_file):
                with open(cls_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(names[i] for i in sorted(names)) + "\n")
    if hdb is not None:
        from AIO_QC.tools.frame_extractor.core.hashdb import phash, thumb_sig
        hdb.add(phash(img), video_key, idx, fname, thumb_sig(img))
    return fname


def _crop(img, roi):
    if not roi:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    return img[y1:y2, x1:x2] if x2 - x1 > 1 and y2 - y1 > 1 else img


class MarkerExtractor:
    """Decoder + optional models for saving / snapping markers (one per video)."""

    def __init__(self, video_path, out_dir, backend="auto", ext=".jpg", detectors=None, preannot="none",
                 det_conf=0.25, hash_db=None, quality=95, name_template=None, on_exists="overwrite"):
        self.video_path = video_path
        self.name = os.path.splitext(os.path.basename(video_path))[0]
        self.out_dir, self.ext = out_dir, ext
        self.detectors, self.preannot, self.det_conf = detectors, preannot, det_conf
        self.src = open_source(video_path, backend, hw_accel=False)
        from AIO_QC.tools.frame_extractor.core.naming import Namer, DEFAULT_TEMPLATE
        self.quality = quality
        self.namer = Namer(name_template or DEFAULT_TEMPLATE, on_exists, video_path, self.src.fps, self.src.time_of)
        self.limit = max(15, int(2 * self.src.fps))
        self.pos = 0
        self.hdb = None
        if hash_db:
            from AIO_QC.tools.frame_extractor.core.hashdb import HashDB
            self.hdb = HashDB(hash_db)

    def read(self, idx):
        if not (self.pos <= idx <= self.pos + self.limit):
            self.src.seek(idx)
            self.pos = idx
        while self.pos < idx:
            if not self.src.grab():
                return None
            self.pos += 1
        ok, img = self.src.read()
        self.pos = idx + 1 if ok else -10 ** 9
        return img if ok else None

    def save(self, idx, roi=None):
        img = self.read(idx)
        if img is None:
            raise IOError(f"could not decode frame {idx}")
        crop = _crop(img, roi)
        fname = save_marked_frame(crop, idx, self.name, self.out_dir, self.ext, self.detectors, self.preannot,
                                  self.det_conf, self.hdb, os.path.abspath(self.video_path), self.quality,
                                  self.namer, roi if crop is not img else None)
        if fname is None:
            raise FileExistsError("exists - skipped (naming policy)")
        return os.path.join(self.out_dir, *fname.split("/")), crop

    def snap(self, idx, radius, roi=None):
        f, s, img = snap_to_sharpest(self.src, idx, radius, roi)
        self.pos = -10 ** 9
        return f, s

    def close(self):
        self.src.release()
        if self.hdb is not None:
            self.hdb.close()
