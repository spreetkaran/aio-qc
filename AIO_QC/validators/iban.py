# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
NAME = "IBAN"
DESCRIPTION = "Country code + 2 check digits + account (mod-97 check)"


def validate(text: str):
    t = text.upper()
    if not (15 <= len(t) <= 34) or not t[:2].isalpha() or not t[2:4].isdigit() or not t.isalnum():
        return False, "not an IBAN shape"
    n = int("".join(str(int(c, 36)) for c in t[4:] + t[:4]))
    return (True, "") if n % 97 == 1 else (False, "IBAN checksum is wrong")
