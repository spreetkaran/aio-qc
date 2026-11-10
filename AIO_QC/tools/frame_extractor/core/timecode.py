# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Timecode formatting and parsing (Qt-free).

Formats
  time       HH:MM:SS.mmm                      wall-clock position (exact for VFR)
  smpte      HH:MM:SS:FF   (HH:MM:SS;FF drop-frame at 29.97 / 59.94)
  frames     frame index
  seconds    123.456 s
  remaining  -HH:MM:SS.mmm                     time left to the end

Every formatter also returns *segments*: (start, end, step) character spans with the
step (in frames or seconds) that the unit represents - the UI uses them to nudge a
unit with the mouse wheel or by dragging, like the timecode fields of an NLE.

parse_position() understands what people type:
  1:23.5   01:02:03.250   00:01:02:15 (SMPTE)   00:01:02;15 (drop-frame)
  1234f  f1234  #1234     frames
  83.2s  83.2  90ms       seconds (plain numbers are seconds, or frames in frames mode)
  42%                     percent of the video
  +5s  -30f  +1:00  -10%  relative to the current position
  start  end  in  out     keywords
"""
import re

FORMATS = ("time", "smpte", "frames", "seconds", "remaining")
FORMAT_LABELS = {"time": "Time", "smpte": "Timecode", "frames": "Frames", "seconds": "Seconds",
                 "remaining": "Remaining"}
FORMAT_CHIPS = {"time": "TIME", "smpte": "TC", "frames": "FR", "seconds": "SEC", "remaining": "REM"}


# --------------------------------------------------------------- SMPTE
def nominal_rate(fps):
    """(integer timebase, drop-frame?) for an fps value: 29.97 -> (30, True), 25 -> (25, False)."""
    if fps <= 0:
        return 30, False
    rb = int(round(fps))
    drop = rb in (30, 60) and abs(fps - rb * 1000 / 1001) < 0.01
    return max(1, rb), drop


def frames_to_smpte(n, fps):
    rb, drop = nominal_rate(fps)
    n = max(0, int(n))
    if drop:
        d = 2 if rb == 30 else 4
        per10 = rb * 600 - d * 9
        per1 = rb * 60 - d
        tens, rem = divmod(n, per10)
        n += d * 9 * tens + (d * ((rem - d) // per1) if rem > d else 0)
    ff = n % rb
    s = n // rb
    sep = ";" if drop else ":"
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}{sep}{ff:02d}"


def smpte_to_frames(hh, mm, ss, ff, fps, drop=None):
    rb, is_drop = nominal_rate(fps)
    drop = is_drop if drop is None else drop and is_drop
    total = ((hh * 3600 + mm * 60 + ss) * rb) + ff
    if drop:
        d = 2 if rb == 30 else 4
        minutes = hh * 60 + mm
        total -= d * (minutes - minutes // 10)
    return total


# --------------------------------------------------------------- formatting
def _hms(seconds, ms=True):
    seconds = max(0.0, seconds)
    t = int(round(seconds * 1000)) if ms else int(seconds) * 1000
    h, t = divmod(t, 3_600_000)
    m, t = divmod(t, 60_000)
    s, t = divmod(t, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}" + (f".{t:03d}" if ms else "")


def format_position(fmt, frame, fps, time_of, duration, frame_count):
    """(text, segments). time_of: frame -> seconds. Segment steps are ('s', seconds) or ('f', frames)."""
    frame = max(0, int(frame))
    if fmt == "smpte":
        rb, _ = nominal_rate(fps)
        # SMPTE counts frames at the nominal rate; for VFR sources derive it from the real time
        n = frame if time_of is None else int(round(time_of(frame) * rb * (1000 / 1001 if nominal_rate(fps)[1] else 1)))
        n = frame if abs(n - frame) <= 1 else n
        text = frames_to_smpte(n, fps)
        return text, [(0, 2, ("s", 3600)), (3, 5, ("s", 60)), (6, 8, ("s", 1)), (9, 11, ("f", 1))]
    if fmt == "frames":
        width = max(1, len(str(max(0, frame_count - 1))))
        text = f"{frame:0{width}d}"
        segs = [(i, i + 1, ("f", 10 ** (len(text) - 1 - i))) for i in range(len(text))]
        return text, segs
    t = time_of(frame) if time_of else frame / (fps or 30.0)
    if fmt == "seconds":
        text = f"{t:.3f}"
        dot = text.index(".")
        segs = [(i, i + 1, ("s", 10 ** (dot - 1 - i))) for i in range(dot)] + [(dot + 1, len(text), ("f", 1))]
        return text + " s", segs
    if fmt == "remaining":
        text = "-" + _hms(max(0.0, duration - t))
        return text, [(1, 3, ("s", -3600)), (4, 6, ("s", -60)), (7, 9, ("s", -1)), (10, 13, ("f", -1))]
    text = _hms(t)
    return text, [(0, 2, ("s", 3600)), (3, 5, ("s", 60)), (6, 8, ("s", 1)), (9, 12, ("f", 1))]


def format_duration(seconds):
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    return _hms(seconds, ms=False) if seconds >= 3600 else _hms(seconds)[3:]


# --------------------------------------------------------------- parsing
_KEYWORDS = ("start", "end", "in", "out")


def parse_position(text, *, current, fps, frame_count, time_of=None, index_of=None, in_frame=None,
                   out_frame=None, mode="time"):
    """Frame index for what the user typed (clamped). Raises ValueError with a readable message."""
    s = text.strip().lower().replace(",", ".")
    if not s:
        raise ValueError("empty")
    last = max(0, frame_count - 1)
    t_of = time_of or (lambda f: f / (fps or 30.0))
    i_of = index_of or (lambda sec: int(round(sec * (fps or 30.0))))
    clamp = lambda f: max(0, min(int(f), last))

    if s in _KEYWORDS:
        v = {"start": 0, "end": last, "in": in_frame, "out": out_frame}[s]
        if v is None:
            raise ValueError(f"no {s} point set")
        return clamp(v)

    rel = 0
    if s[0] in "+-" and len(s) > 1:
        rel = 1 if s[0] == "+" else -1
        s = s[1:].strip()

    # percent
    if s.endswith("%"):
        p = float(s[:-1]) / 100.0
        if rel:
            return clamp(current + rel * p * last)
        return clamp(p * last)

    # explicit frames
    m = re.fullmatch(r"(?:f|#)\s*(\d+)|(\d+)\s*f(?:r|rames?)?", s)
    if m:
        n = int(m.group(1) or m.group(2))
        return clamp(current + rel * n if rel else n)

    # SMPTE HH:MM:SS:FF or HH:MM:SS;FF
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2})([:;])(\d{1,3})", s)
    if m:
        hh, mm, ss, sep, ff = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
        rb, _ = nominal_rate(fps)
        if ff >= rb:
            raise ValueError(f"frame field must be < {rb}")
        n = smpte_to_frames(hh, mm, ss, ff, fps, drop=(sep == ";") or None)
        if rel:
            return clamp(current + rel * n)
        return clamp(n)

    # clock time [[H:]M:]S[.fff]
    m = re.fullmatch(r"(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d*)?)", s)
    if m and ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return _from_seconds(sec, rel, current, t_of, i_of, clamp)

    # seconds / milliseconds / bare numbers
    m = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+)\s*(ms|s|sec|secs|seconds?|m|min|h)?", s)
    if m:
        v, unit = float(m.group(1)), m.group(2)
        if unit is None and mode == "frames" and "." not in m.group(1):
            return clamp(current + rel * int(v) if rel else int(v))
        mult = {"ms": 0.001, "m": 60, "min": 60, "h": 3600}.get(unit, 1.0)
        return _from_seconds(v * mult, rel, current, t_of, i_of, clamp)

    raise ValueError(f"can't read '{text}' - try 1:23.5, 00:01:23:12, 1234f, 42%, +5s")


def _from_seconds(sec, rel, current, t_of, i_of, clamp):
    if rel:
        return clamp(i_of(max(0.0, t_of(current) + rel * sec)))
    return clamp(i_of(sec))


def nudge(frame, step, amount, fps, time_of=None, index_of=None, frame_count=0):
    """Move by `amount` units of a segment step ('f', n) or ('s', seconds)."""
    kind, v = step
    last = max(0, frame_count - 1) if frame_count else None
    if kind == "f":
        out = frame + int(v * amount)
    else:
        t_of = time_of or (lambda f: f / (fps or 30.0))
        i_of = index_of or (lambda s: int(round(s * (fps or 30.0))))
        out = i_of(max(0.0, t_of(frame) + v * amount))
        if out == frame and amount:       # sub-frame step at low fps: move at least one frame
            out = frame + (1 if v * amount > 0 else -1)
    out = max(0, out)
    return min(out, last) if last is not None else out
