# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
NAME = "Barcode number (EAN-8/13, UPC-A, GTIN-14)"
DESCRIPTION = "8, 12, 13 or 14 digits with a valid GS1 check digit"


def validate(text: str):
    t = "".join(c for c in text if c.isdigit())
    if len(t) not in (8, 12, 13, 14) or len(t) != len(text):
        return False, "needs 8, 12, 13 or 14 digits only"
    body, check = t[:-1], int(t[-1])
    s = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    exp = (10 - s % 10) % 10
    return (True, "") if exp == check else (False, f"check digit should be {exp}")
