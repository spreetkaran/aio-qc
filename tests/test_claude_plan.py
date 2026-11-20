# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Claude Review through Claude Code on this computer (the user's own sign-in and plan), with a stand-in `claude`
program: the structured-output call (`-p`, `--json-schema`, the images as files it reads), API-key variables kept out
of its environment, findings and Apply as with the API, the per-run image limit, no batches, a signed-out Claude Code
stopping the run with its own message, and the connection test."""
import json
import os
import sys
from pathlib import Path

import pytest

from .test_aio_image import _open, shell  # noqa: F401
from .test_claude_review import _ds, _run

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in program is a Python script")

FAKE = r'''#!PY
import json, os, re, sys
args = sys.argv[1:]
log = os.environ.get("FAKE_CC_LOG")
if "--version" in args:
    print("9.9.9 (Claude Code)"); sys.exit(0)
def out(d, code=0):
    print(json.dumps(d)); sys.exit(code)
if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
    out({"type": "result", "subtype": "success", "is_error": True, "result": "an API key was used"}, 1)
if os.environ.get("FAKE_CC_MODE") == "logout":
    out({"type": "result", "subtype": "success", "is_error": True, "result": "Not logged in · Please run /login"}, 1)
prompt = args[args.index("-p") + 1]
stdin = sys.stdin.read()
if log:
    open(log, "a").write(json.dumps({"args": args, "stdin": stdin, "cwd": os.getcwd()}) + "\n")
if "--json-schema" not in args:
    out({"type": "result", "subtype": "success", "is_error": False, "result": "OK"})
assert args[args.index("--allowedTools") + 1] == "Read" and args[args.index("--permission-mode") + 1] == "dontAsk"
json.loads(args[args.index("--json-schema") + 1])
paths = re.findall(r"^- (\S+\.jpg) — ", stdin, re.M)
assert paths and all(os.path.isfile(p) for p in paths), stdin
import cv2, numpy as np
img = cv2.imread(paths[-1])
found = []
if "#2 red" in stdin:
    found.append({"type": "wrong_class", "shape": 2, "suggested_class": "blue", "confidence": 0.9,
                  "reason": "the square is blue"})
patch = img[95:105, 110:130]
if (patch[:, :, 2] > 180).mean() > 0.5 and (patch[:, :, 0] < 80).mean() > 0.5:
    found.append({"type": "missing_object", "shape": None, "suggested_class": "red", "box": [100, 90, 140, 110],
                  "confidence": 0.8, "reason": "an unlabelled red mark"})
out({"type": "result", "subtype": "success", "is_error": False, "result": "", "structured_output":
     {"findings": found, "image_ok": not found}, "usage": {"input_tokens": 10, "cache_read_input_tokens": 900,
                                                          "output_tokens": 40}, "total_cost_usd": 0.0})
'''


@pytest.fixture
def cc(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    p = d / "claude"
    p.write_text(FAKE.replace("#!PY", "#!" + sys.executable))
    p.chmod(0o755)
    log = tmp_path / "cc.log"
    monkeypatch.setenv("FAKE_CC_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ.get('PATH', '')}")
    return p, log


def _calls(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def _window(shell, qtbot, root, scope="all"):  # noqa: F811
    w = _open(shell, qtbot, root)
    w.settings.setValue("claude/backend", "claude_code")
    w.settings.setValue("claude/scope", scope)
    w.settings.setValue("claude/how", "live")
    w.settings.setValue("claude/cc_path", "")
    w.open_tool("claude_review")
    return w, w.open_dialogs["claude_review"]


def test_plan_review_apply_and_limits(shell, qtbot, tmp_path, cc, monkeypatch):  # noqa: F811
    from AIO_QC.core import claude_review as CR
    prog, log = cc
    assert CR.find_claude() == str(prog)
    root = _ds(tmp_path)
    w, d = _window(shell, qtbot, root)
    assert d.backend.currentData() == "claude_code" and "your plan" in d.conn_note.text()
    assert not d.how.model().item(d.how.findData("batch")).isEnabled()           # no batches on a plan
    _run(d, qtbot)
    calls = _calls(log)
    assert len(calls) == 10, (d.status.text(), d.errors)
    a = calls[0]["args"]
    assert a[a.index("--model") + 1] == "claude-sonnet-5-5" and "--bare" not in a
    assert "#1 red — rectangle, box [10, 20, 60, 70]" in calls[0]["stdin"] and "- red: a red square" in \
        calls[0]["stdin"]
    assert not Path(calls[0]["cwd"]).exists()                                     # its temporary folder is gone
    got = sorted((Path(r.key).name, r.cells[0], r.shape_index) for r in d.rows)
    assert got == [("im0.json", "Missing object", None), ("im4.json", "Wrong class", 1)], (got, d.errors)
    assert "with Claude Code (your plan)" in d.summarize() and "(9,100 / 400 tokens" in d.summarize()
    # Apply works the same way
    from PyQt6.QtCore import Qt
    for t in range(d.table.rowCount()):
        d.table.item(t, 0).setCheckState(Qt.CheckState.Checked if d._row_at(t).key.endswith("im4.json") else
                                         Qt.CheckState.Unchecked)
    d._apply()
    from .test_similarity_fake import _labels
    assert _labels(root, "im4.json") == ["red", "blue"]
    d.close()
    # at most CC_PLAN_LIMIT images per run (asked; the first N)
    monkeypatch.setattr(CR, "CC_PLAN_LIMIT", 3)
    root2 = _ds(tmp_path / "b")
    w2, d2 = _window(shell, qtbot, root2)
    n0 = len(_calls(log))
    _run(d2, qtbot)
    assert len(_calls(log)) - n0 == 3 and "3 reviewed now" in d2.summarize()
    d2.close()


def test_plan_signed_out_and_connection(shell, qtbot, tmp_path, cc, monkeypatch):  # noqa: F811
    prog, log = cc
    root = _ds(tmp_path)
    w, d = _window(shell, qtbot, root)
    monkeypatch.setenv("FAKE_CC_MODE", "logout")
    _run(d, qtbot)
    assert any("Not logged in" in e for e in d.errors) and "stopped: Not logged in" in d.summarize()
    assert not d.rows
    monkeypatch.delenv("FAKE_CC_MODE")
    from AIO_QC.widgets.claude_review_dialog import ConnectionDialog
    c = ConnectionDialog(d, w, d.store)
    c._test_cc()
    qtbot.waitUntil(lambda: c.cc_lbl.text().startswith(("✓", "✗")), timeout=20000)
    assert c.cc_lbl.text().startswith("✓ 9.9.9 (Claude Code)") and "answered: OK" in c.cc_lbl.text()
    # not installed: a clear message, nothing runs
    c.cc_path.setText(str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr("AIO_QC.core.claude_review.Path.home", lambda: tmp_path / "nohome")
    c._test_cc()
    assert "not found" in c.cc_lbl.text()
    c.reject()
    d.close()
