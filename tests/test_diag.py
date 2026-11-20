# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Errors handled by carrying on leave one short, capped, secret-free line in the log."""
from loguru import logger

from AIO_QC.core import diag


def test_ignored_is_capped_short_and_masked():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG", format="{message}")
    try:
        for i in range(1200):
            try:
                raise ValueError(f"bad file {i} api_key=sk-THISISSECRET123456 token: abcdefghijk")
            except ValueError as exc:
                diag.ignored(exc, "thumbnail")
    finally:
        logger.remove(sink)
    assert len(lines) == 3 + 3                       # the first 3, then the 10th, 100th, 1000th
    assert lines[0].startswith("ignored (thumbnail) ValueError: bad file 0")
    assert "[×1000 here]" in lines[-1] and "Traceback" not in "".join(lines)
    assert "THISISSECRET" not in "".join(lines) and "abcdefghijk" not in "".join(lines)
    assert max(diag.counts().values()) >= 1200


def test_the_log_file_keeps_no_variable_values():
    import inspect
    from AIO_QC import __main__ as m
    assert "diagnose=False" in inspect.getsource(m._setup_loguru)


def test_exit_report_lists_busy_handlers_and_dumps(tmp_path, monkeypatch):
    import json
    for _ in range(3):
        try:
            raise KeyError("x")
        except KeyError as exc:
            diag.ignored(exc)
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG", format="{message}")
    out = tmp_path / "d.json"
    monkeypatch.setenv("AIOQC_DIAG_DUMP", str(out))
    try:
        diag.report()
    finally:
        logger.remove(sink)
    assert lines[0].startswith("handled errors this session") and any("KeyError ×3" in x for x in lines)
    assert any(v.get("KeyError") == 3 for v in json.loads(out.read_text()).values())
