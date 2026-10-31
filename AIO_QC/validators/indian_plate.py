# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
NAME = "Indian number plate"
DESCRIPTION = ("State + RTO + series + 4-digit number (KA01AB1234, DL3CAB1234; HSRP plates print leading zeros), "
               "BH series (22BH1234AA); one or two lines joined")

import re

_STATES = {"AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
           "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
           "UP", "WB", "CT", "TG", "UA"}
_REGULAR = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")
_BH = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def validate(text: str):
    t = text.upper()
    if _BH.match(t):
        return True
    m = _REGULAR.match(t)
    if not m:
        digits = len(t) - len(t.rstrip("0123456789"))
        if t[:2].isalpha() and 0 < digits < 4:
            return False, f"number has {digits} digit(s) — 4 expected (truncated box?)"
        return False, "not STATE + RTO + SERIES + NUMBER or a BH-series plate"
    if m.group(1) not in _STATES:
        return False, f"unknown state code {m.group(1)}"
    return True
