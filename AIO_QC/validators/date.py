# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
NAME = "Date"
DESCRIPTION = "A real calendar date: DDMMYYYY, DDMMYY, YYYYMMDD (separators are removed by normalisation)"

from datetime import date


def _ok(y, m, d):
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def validate(text: str):
    t = text
    if not t.isdigit():
        return False, "digits only expected"
    if len(t) == 8 and (_ok(int(t[4:]), int(t[2:4]), int(t[:2])) or _ok(int(t[:4]), int(t[4:6]), int(t[6:]))):
        return True
    if len(t) == 6 and _ok(2000 + int(t[4:]), int(t[2:4]), int(t[:2])):
        return True
    return False, "not a valid date"
