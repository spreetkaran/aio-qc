# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Exact per-frame timestamps and keyframes for MP4 / MOV / M4V / 3GP - from the file's index, in milliseconds.

Phones and most dashcams record variable frame rate: frame durations wander (e.g. "29.98 fps"). Treating such
a video as constant-rate maps frame numbers to the wrong times (and seeks to the wrong frames). The container
already stores every frame's timing in the `moov` box:

    stts  decode-time deltas      ctts  composition offsets (B-frame reordering)     stss  keyframes
    mdhd  timescale               elst  edit list (presentation start offset)

Only the index is read (a few MB at most, wherever `moov` sits in the file) - no decoding, no packet scan,
so a 5-hour video is as fast as a 5-second one. Returns None for fragmented MP4 or anything unexpected; callers
then fall back to PyAV / constant-rate timing.
"""
import os
import struct

import numpy as np

MP4_EXTS = (".mp4", ".mov", ".m4v", ".3gp", ".3g2", ".mj2")
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


def _boxes(buf, start, end):
    """Yield (type, payload_start, payload_end) for boxes in buf[start:end]."""
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, i)
        hdr = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack_from(">Q", buf, i + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            return
        yield typ, i + hdr, i + size
        i += size


def _read_moov(path):
    """Locate the top-level moov box without reading media data."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        fsize = f.tell()
        pos = 0
        moof = False
        while pos + 8 <= fsize:
            f.seek(pos)
            hdr = f.read(16)
            if len(hdr) < 8:
                break
            size, typ = struct.unpack_from(">I4s", hdr, 0)
            hl = 8
            if size == 1:
                size = struct.unpack_from(">Q", hdr, 8)[0]
                hl = 16
            elif size == 0:
                size = fsize - pos
            if size < hl:
                break
            if typ == b"moof":
                moof = True
            if typ == b"moov":
                if size > 512 * 2 ** 20:          # absurd index size: not worth it
                    return None, moof
                f.seek(pos)
                return f.read(size), moof
            pos += size
    return None, moof


def _full_box(buf, s):
    return buf[s], s + 4          # version, payload after version/flags


def _parse_trak(buf, s, e):
    info = {}
    for typ, ps, pe in _boxes(buf, s, e):
        if typ == b"edts":
            for t2, s2, e2 in _boxes(buf, ps, pe):
                if t2 == b"elst":
                    v, p = _full_box(buf, s2)
                    n = struct.unpack_from(">I", buf, p)[0]
                    p += 4
                    edits = []
                    for _ in range(n):
                        if v == 1:
                            dur, mt = struct.unpack_from(">Qq", buf, p)
                            p += 16
                        else:
                            dur, mt = struct.unpack_from(">Ii", buf, p)
                            p += 8
                        p += 4                    # media rate
                        edits.append((dur, mt))
                    info["elst"] = edits
        elif typ == b"mdia":
            for t2, s2, e2 in _boxes(buf, ps, pe):
                if t2 == b"mdhd":
                    v, p = _full_box(buf, s2)
                    info["timescale"] = struct.unpack_from(">I", buf, p + (16 if v == 1 else 8))[0]
                elif t2 == b"hdlr":
                    info["handler"] = bytes(buf[s2 + 8:s2 + 12])
                elif t2 == b"minf":
                    for t3, s3, e3 in _boxes(buf, s2, e2):
                        if t3 == b"stbl":
                            for t4, s4, e4 in _boxes(buf, s3, e3):
                                if t4 in (b"stts", b"ctts", b"stss", b"stsz"):
                                    info[t4.decode()] = (s4, e4)
    return info


def _table(buf, rng, fields, signed=False):
    s, e = rng
    v, p = _full_box(buf, s)
    n = struct.unpack_from(">I", buf, p)[0]
    p += 4
    dt = np.dtype(">i4" if (signed or v == 1) and fields == 2 else ">u4")
    if fields == 2:
        arr = np.frombuffer(buf, dtype=np.dtype([("a", ">u4"), ("b", dt)]), count=n, offset=p)
        return arr["a"].astype(np.int64), arr["b"].astype(np.int64)
    return np.frombuffer(buf, dtype=">u4", count=n, offset=p).astype(np.int64)


def mp4_index(path):
    """(timestamps_seconds [presentation order, first = 0], keyframe indices) or None."""
    if not path.lower().endswith(MP4_EXTS):
        return None
    try:
        moov, fragmented = _read_moov(path)
        if moov is None or fragmented:
            return None
        buf = memoryview(moov)
        video = None
        for typ, s, e in _boxes(buf, 8, len(buf)):
            if typ == b"trak":
                t = _parse_trak(buf, s, e)
                if t.get("handler") == b"vide" and "stts" in t and "timescale" in t:
                    video = t
                    break
        if video is None:
            return None
        counts, deltas = _table(buf, video["stts"], 2)
        n = int(counts.sum())
        if n < 2 or n > 50_000_000:
            return None
        dts = np.concatenate([[0], np.cumsum(np.repeat(deltas, counts))[:-1]])
        pts = dts.copy()
        if "ctts" in video:
            cc, co = _table(buf, video["ctts"], 2, signed=True)
            off = np.repeat(co, cc)
            if len(off) != n:
                return None
            pts = dts + off
        order = np.argsort(pts, kind="stable")          # decode order -> presentation order
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n)
        sorted_pts = pts[order]
        ts = (sorted_pts - sorted_pts[0]) / float(video["timescale"])
        if "stss" in video:
            sync = _table(buf, video["stss"], 1) - 1            # 1-based decode-order sample numbers
            sync = sync[(sync >= 0) & (sync < n)]
            keys = np.sort(rank[sync]).tolist()
        else:
            keys = list(range(n))                              # no stss: every sample is a sync sample
        return ts.tolist(), keys
    except (struct.error, ValueError, OSError, IndexError):
        return None
