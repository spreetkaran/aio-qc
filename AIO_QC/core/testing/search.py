# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame search: find moments in videos by a text description ("red truck at night") or by example
("frames like this one / like this ROI").

Embedders
  clip     CLIP (open_clip or Hugging Face transformers) inside a Python environment, through the same
           inference server as the YOLO models (frames via shared memory). Text AND image queries.
           Weights download once, in that environment.
  classic  built in, no install, offline: colour (HSV histogram), layout (exposure-normalised 16x12 grey) and
           edge-orientation features. Image queries only ("find similar"); good for near-identical scenes,
           not for meaning.

Index per (video, embedder): frames sampled every N seconds with ONE ffmpeg process (GPU decode where
available, scaled to 384 px in ffmpeg), embedded in batches, stored in ~/.frame_extractor/search/ with a
small thumbnail per sample. Interrupted builds resume where they stopped. Results are spread in time
(a moment is not returned 20 times).
"""
import hashlib
import json
import os
import subprocess
import time

import cv2
import numpy as np

SAMPLE_W = 384
THUMB_W = 192


def search_dir():
    from AIO_QC.core.fx_storage import fx_home
    d = str(fx_home() / "search")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ embedders
class ClassicEmbedder:
    id = "classic"
    text = False
    label = "Built-in (find similar)"

    def embed(self, frames):
        out = []
        for f in frames:
            small = cv2.resize(f, (128, 96), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            h = np.sqrt(h / (h.sum() + 1e-9))
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lay = cv2.resize(g, (16, 12), interpolation=cv2.INTER_AREA).ravel()
            lay = (lay - lay.mean()) / (lay.std() + 4.0)
            gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            bins = (ang / (2 * np.pi) * 8).astype(int) % 8
            hog = np.zeros((4, 4, 8), np.float32)
            for cy in range(4):
                for cx in range(4):
                    sl = (slice(cy * 24, (cy + 1) * 24), slice(cx * 32, (cx + 1) * 32))
                    hog[cy, cx] = np.bincount(bins[sl].ravel(), mag[sl].ravel(), 8)
            hog = hog.ravel()
            hog = hog / (np.linalg.norm(hog) + 1e-9)
            v = np.concatenate([h / (np.linalg.norm(h) + 1e-9) * 1.0, lay / (np.linalg.norm(lay) + 1e-9) * 0.8,
                                hog * 0.8])
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.asarray(out, np.float32)

    def embed_text(self, texts):
        raise RuntimeError("Text search needs a CLIP model (Settings in the search panel)")


class ClipEmbedder:
    text = True

    def __init__(self, python, model="ViT-B-32", pretrained="laion2b_s34b_b79k", backend="auto", device="auto",
                 hf_model="openai/clip-vit-base-patch32"):
        from AIO_QC.tools.frame_extractor.core.detector import _client
        self.spec = dict(backend=backend, model=model, pretrained=pretrained, device=device, hf_model=hf_model)
        self.client = _client(python)
        self.info = self.client.embed_load(self.spec)
        self.id = "clip-" + hashlib.sha1(self.info["model"].encode()).hexdigest()[:8]
        self.label = f"CLIP · {self.info['model']} ({self.info['backend']}, {self.info['device']})"

    def embed(self, frames, cancel=None):
        return np.asarray(self.client.embed_frames(frames, cancel), np.float32)

    def embed_text(self, texts):
        # prompt ensemble: the raw words and "a photo of ..." averaged - steadier than either alone
        prompts = []
        for t in texts:
            prompts += [t, f"a photo of {t}", f"a frame from a video of {t}"]
        v = np.asarray(self.client.embed_text(prompts), np.float32).reshape(len(texts), 3, -1).mean(1)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------ sampling (one ffmpeg process)
def sample_frames(video, interval, ffmpeg, start=0.0, width=SAMPLE_W, cancel=None):
    """Yield (time, BGR frame width x ...) every `interval` seconds from `start`."""
    from AIO_QC.core import procs
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    src = open_source(video, "auto", hw_accel=False)
    w, h = src.width, src.height
    src.release()
    ow = min(width, w) // 2 * 2
    oh = max(2, int(round(h * ow / w)) // 2 * 2)
    fb = ow * oh * 3
    for hw in (["-hwaccel", "auto"], []):
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *hw]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", video, "-an", "-sn", "-vf", f"fps=1/{interval:.6f}:round=near,scale={ow}:{oh}:flags=area",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        proc = procs.popen(cmd, background=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, bufsize=fb * 4)
        k, got = 0, False
        try:
            while True:
                if cancel is not None and cancel():
                    return
                buf = proc.stdout.read(fb)
                if len(buf) < fb:
                    break
                got = True
                yield start + k * interval, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
                k += 1
        finally:
            proc.kill()
            proc.wait()
        if got:
            return


# ------------------------------------------------------------------ index
class VideoIndex:
    def __init__(self, video, embedder_id):
        from AIO_QC.core.testing.runner import video_key
        self.video = os.path.abspath(video)
        self.embedder_id = embedder_id
        self.path = os.path.join(search_dir(), f"{video_key(video)}_{embedder_id}.npz")
        self.times = np.zeros(0, np.float32)
        self.vecs = None
        self.thumbs = []
        self.complete = False
        self.interval = None

    def load(self):
        if not os.path.isfile(self.path):
            return False
        try:
            d = np.load(self.path, allow_pickle=False)
            meta = json.loads(str(d["meta"]))
            self.times, self.vecs = d["times"], d["vecs"].astype(np.float32)
            blob, offs = d["thumbs"].tobytes(), d["offs"]
            self.thumbs = [blob[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
            self.complete, self.interval = meta["complete"], meta["interval"]
            return True
        except Exception:
            return False

    def save(self):
        offs = np.cumsum([0] + [len(t) for t in self.thumbs]).astype(np.int64)
        blob = np.frombuffer(b"".join(self.thumbs), np.uint8) if self.thumbs else np.zeros(0, np.uint8)
        meta = json.dumps(dict(video=self.video, complete=self.complete, interval=self.interval, saved=time.time()))
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, times=self.times, vecs=(self.vecs if self.vecs is not None else np.zeros((0, 1))).astype(np.float16),
                 thumbs=blob, offs=offs, meta=np.array(meta))
        os.replace(tmp, self.path)

    def build(self, embedder, ffmpeg, interval=None, batch=32, progress=None, cancel=None):
        """(Re)build or resume. interval None = 1 s; long videos get sparser sampling (<= ~6000 samples)."""
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        progress = progress or (lambda *a: None)
        src = open_source(self.video, "auto", hw_accel=False)
        dur = src.duration()
        src.release()
        interval = max(float(interval or 1.0), round(dur / 6000.0, 1))
        if self.load() and self.interval == interval:
            if self.complete:
                return self
            start = float(self.times[-1]) + interval if len(self.times) else 0.0
        else:
            self.times, self.vecs, self.thumbs, start = np.zeros(0, np.float32), None, [], 0.0
        self.interval = interval
        total = max(1, int(dur / interval) + 1)
        times, frames = list(self.times), []
        vecs = [self.vecs] if self.vecs is not None and len(self.vecs) else []
        pend_t, last_save = [], time.time()

        def flush():
            nonlocal frames, pend_t, last_save
            if not frames:
                return
            v = embedder.embed(frames)
            vecs.append(np.asarray(v, np.float32))
            self.thumbs += [cv2.imencode(".jpg", cv2.resize(f, (THUMB_W, int(f.shape[0] * THUMB_W / f.shape[1])),
                                                            interpolation=cv2.INTER_AREA),
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes() for f in frames]
            times.extend(pend_t)
            frames, pend_t = [], []
            if time.time() - last_save > 20:                  # crash-safe: resume from here
                self.times, self.vecs = np.asarray(times, np.float32), np.concatenate(vecs)
                self.save()
                last_save = time.time()
        for t, f in sample_frames(self.video, interval, ffmpeg, start, cancel=cancel):
            frames.append(f.copy())
            pend_t.append(t)
            if len(frames) >= batch:
                flush()
                progress(len(times), total, f"{os.path.basename(self.video)} · {t:.0f} s")
        if not (cancel and cancel()):
            flush()
            self.complete = True
        self.times = np.asarray(times, np.float32)
        self.vecs = np.concatenate(vecs) if vecs else np.zeros((0, 1), np.float32)
        self.save()
        progress(len(times), total, "done")
        return self


def search(indexes, query, k=24, min_gap=2.0, negative=None, exclude=None):
    """indexes: [VideoIndex]; query: unit vector (or several, averaged). Returns
    [dict(video, time, score, thumb)] best first, at least min_gap seconds apart within a video."""
    q = np.asarray(query, np.float32).reshape(-1, np.asarray(query).shape[-1]).mean(0)
    q /= np.linalg.norm(q) + 1e-12
    cands = []
    for ix in indexes:
        if ix.vecs is None or not len(ix.vecs) or ix.vecs.shape[1] != q.shape[0]:
            continue
        s = ix.vecs @ q
        if negative is not None:
            nv = np.asarray(negative, np.float32).reshape(-1, q.shape[0])
            s = s - 0.5 * np.clip(ix.vecs @ nv.T, 0, None).max(1)
        top = np.argsort(-s)[:k * 20]
        cands += [(float(s[i]), ix, int(i)) for i in top]
    cands.sort(key=lambda c: -c[0])
    out, taken = [], {}
    for score, ix, i in cands:
        t = float(ix.times[i])
        if exclude and exclude[0] == ix.video and abs(exclude[1] - t) < min_gap:
            continue
        prev = taken.setdefault(ix.video, [])
        if any(abs(t - p) < min_gap for p in prev):
            continue
        prev.append(t)
        out.append(dict(video=ix.video, time=t, score=score, thumb=ix.thumbs[i] if i < len(ix.thumbs) else b""))
        if len(out) >= k:
            break
    return out
