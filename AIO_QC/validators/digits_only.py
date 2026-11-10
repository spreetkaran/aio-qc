# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
NAME = "Digits only"
DESCRIPTION = "One or more digits (meter readings, counters, serials)"


def validate(text: str):
    return (True, "") if text.isdigit() else (False, "contains non-digits")
