# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
NAME = "ISBN (10 or 13)"
DESCRIPTION = "ISBN-10 (last may be X) or ISBN-13 with a valid check digit"


def validate(text: str):
    t = text.upper()
    if len(t) == 10 and t[:9].isdigit() and (t[9].isdigit() or t[9] == "X"):
        s = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(t))
        return (True, "") if s % 11 == 0 else (False, "ISBN-10 check digit is wrong")
    if len(t) == 13 and t.isdigit():
        s = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(t[:12]))
        return (True, "") if (10 - s % 10) % 10 == int(t[12]) else (False, "ISBN-13 check digit is wrong")
    return False, "needs 10 or 13 characters"
