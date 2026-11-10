# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
NAME = "Letters and digits"
DESCRIPTION = "A–Z and 0–9 only, at least 2 characters (codes, IDs, part numbers)"


def validate(text: str):
    if len(text) < 2:
        return False, "too short"
    return (True, "") if text.isalnum() and text.isascii() else (False, "contains symbols or non-Latin characters")
