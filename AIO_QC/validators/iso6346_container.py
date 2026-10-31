# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
NAME = "Container ID (ISO 6346)"
DESCRIPTION = "Owner code (3 letters) + category (U/J/Z) + 6-digit serial + check digit, e.g. MSKU1234565"

_VAL = {**{str(d): d for d in range(10)},
        **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38]))}


def validate(text: str):
    t = text.upper()
    if len(t) != 11:
        return False, f"{len(t)} characters (needs 11)"
    if not (t[:4].isalpha() and t[4:].isdigit()):
        return False, "needs 4 letters then 7 digits"
    if t[3] not in "UJZ":
        return False, f"4th letter must be U, J or Z (is {t[3]})"
    total = sum(_VAL[c] * (2 ** i) for i, c in enumerate(t[:10]))
    check = total % 11 % 10
    if check != int(t[10]):
        return False, f"check digit should be {check}, is {t[10]}"
    return True
