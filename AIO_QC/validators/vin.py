# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
NAME = "VIN (vehicle identification number)"
DESCRIPTION = "17 characters, no I/O/Q; the North-American check digit (position 9) is checked when it is a digit/X"

_T = dict(zip("ABCDEFGHJKLMNPRSTUVWXYZ", [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9]))
_W = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def validate(text: str):
    t = text.upper()
    if len(t) != 17:
        return False, f"{len(t)} characters (needs 17)"
    if any(c in "IOQ" for c in t) or not t.isalnum():
        return False, "letters I, O, Q and symbols are not allowed"
    try:
        s = sum((int(c) if c.isdigit() else _T[c]) * w for c, w in zip(t, _W))
    except KeyError:
        return False, "invalid character"
    check = "X" if s % 11 == 10 else str(s % 11)
    if t[8] in "0123456789X" and t[8] != check:
        return False, f"check digit (9th) should be {check}"
    return True
