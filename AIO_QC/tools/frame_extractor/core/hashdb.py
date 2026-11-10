# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Cross-video near-duplicate memory.

A 64-bit perceptual hash (pHash: DCT of a 32x32 grayscale) of every saved
frame goes into a SQLite file shared by all videos, runs and worker processes
of a dataset (default: <output root>/.frameextractor_hashes.db). Before saving,
a frame is compared (Hamming distance) against everything already extracted,
so re-running a camera next week does not re-extract the same scenes.

Rejected frames (moved out in the review gallery) keep their hash, so they are
not extracted again either.
"""
import os
import sqlite3
import time

import cv2
import numpy as np

_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def phash(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int(np.packbits(bits).view(">u8")[0])


def _to_signed(h):
    return h - (1 << 64) if h >= (1 << 63) else h


def _popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)


class HashDB:
    def __init__(self, path, max_dist=6, refresh_every=5.0):
        self.path = path
        self.max_dist = int(max_dist)
        self.refresh_every = refresh_every
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.con = sqlite3.connect(path, timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS hashes (id INTEGER PRIMARY KEY, h INTEGER NOT NULL, "
                         "video TEXT, frame INTEGER, file TEXT, added REAL)")
        cols = {r[1] for r in self.con.execute("PRAGMA table_info(hashes)")}
        if "sig" not in cols:       # v0.12: small exposure-normalised thumbnail to verify hash matches
            self.con.execute("ALTER TABLE hashes ADD COLUMN sig BLOB")
        self.con.commit()
        self._sigs = []
        self._h = np.zeros(0, np.uint64)
        self._keys = []          # (video, frame) per row, for self-match exclusion
        self._last_id = 0
        self._t = 0.0
        self.refresh(force=True)

    def refresh(self, force=False):
        """Pick up hashes added by other workers since the last look."""
        now = time.monotonic()
        if not force and now - self._t < self.refresh_every:
            return
        self._t = now
        rows = self.con.execute("SELECT id, h, video, frame, sig FROM hashes WHERE id > ? ORDER BY id",
                                (self._last_id,)).fetchall()
        if rows:
            self._last_id = rows[-1][0]
            new = np.array([r[1] for r in rows], np.int64).view(np.uint64)
            self._h = np.concatenate([self._h, new])
            self._keys.extend((r[2], r[3]) for r in rows)
            self._sigs.extend(r[4] for r in rows)

    def find(self, h, video=None, frame=None, sig=None, global_pct=2.0, local_pct=8.0):
        """Closest earlier frame within max_dist that is not this very frame (so resumed runs don't flag their
        own frames) AND - when both have a stored thumbnail - passes the local check: a fixed camera's hash is
        dominated by the background, so two different cars at the same gate can share a hash; the local
        comparison tells them apart. Returns (dist, key) or None."""
        self.refresh()
        if not len(self._h):
            return None
        d = _popcount64(self._h ^ np.uint64(h))
        idx = np.nonzero(d <= self.max_dist)[0]
        for i in idx[np.argsort(d[idx])]:
            if self._keys[i] == (video, frame):
                continue
            ref = self._sigs[i] if i < len(self._sigs) else None
            if sig is not None and ref is not None:
                g, l = thumb_compare(sig, ref)
                if g >= global_pct or l >= local_pct:
                    continue                          # same background, different content
            return int(d[i]), self._keys[i]
        return None

    def add(self, h, video, frame, file, sig=None):
        self.con.execute("INSERT INTO hashes (h, video, frame, file, added, sig) VALUES (?,?,?,?,?,?)",
                         (_to_signed(h), video, int(frame), file, time.time(), sig))
        self.con.commit()
        self._h = np.append(self._h, np.uint64(h))
        self._keys.append((video, int(frame)))
        self._sigs.append(sig)
        self._last_id = max(self._last_id, self.con.execute("SELECT last_insert_rowid()").fetchone()[0])

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]

    def clear(self):
        self.con.execute("DELETE FROM hashes")
        self.con.commit()
        self._h = np.zeros(0, np.uint64)
        self._keys, self._last_id = [], 0
        self._sigs = []

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


def default_db_path(output_root):
    return os.path.join(output_root, ".frameextractor_hashes.db")


THUMB = (48, 27)        # stored per frame: 1.3 kB


def thumb_sig(img):
    """Exposure-normalised 48 x 27 greyscale thumbnail (bytes) for verifying hash matches."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    t = cv2.resize(g, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = cv2.resize(t, THUMB, interpolation=cv2.INTER_AREA)
    m, sd = float(t.mean()), float(t.std())
    n = np.clip((t - m) / max(sd, 4.0) * 50.0 + 128.0, 0, 255).astype(np.uint8)
    return n.tobytes()


def thumb_compare(a, b):
    """(global %, local %) between two stored thumbnails; local = largest 3 x 3 cell difference."""
    x = np.frombuffer(a, np.uint8).reshape(THUMB[1], THUMB[0]).astype(np.float32)
    y = np.frombuffer(b, np.uint8).reshape(THUMB[1], THUMB[0]).astype(np.float32)
    d = np.abs(x - y)
    cells = d.reshape(9, 3, 16, 3).mean(axis=(1, 3))
    return float(d.mean()) * 100 / 255, float(cells.max()) * 100 / 255
